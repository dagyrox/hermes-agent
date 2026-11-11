"""HTTP + credential layer for the OCC read-only Kanban contract.

Three static, GET-only endpoints:

    GET /api/plugins/kanban/occ/v1/summary
    GET /api/plugins/kanban/occ/v1/tasks
    GET /api/plugins/kanban/occ/v1/task?task_id=t_xxxxxxxx

Everything security-relevant lives here rather than in the projection module:

* **Route-scoped credential.** The three paths register with their own verifier
  on the generic token-auth seam, so the OCC read token can never authenticate
  on another token route (notably ``/api/gateway/drain``). The token is a
  dedicated environment variable, compared in constant time over SHA-256
  digests, and never enters the global provider stack.
* **Machine-only callers.** Cookies, session headers, ``?token=`` query
  credentials and every forwarding header are refused rather than ignored:
  behind a same-host proxy every peer looks like loopback, so "ignore the
  header" would protect nothing.
* **Private peers only.** Loopback, RFC1918 and ULA addresses; the peer comes
  from ``scope["client"]`` (never ``X-Forwarded-For``).
* **Handler identity re-check.** Each handler re-verifies, by object identity,
  that it was reached by the route-scoped principal. A cookie session that
  arrives through an unregistered alias path gets 404 here.
* **Hand-written query parsing.** FastAPI's ``Query`` machinery silently keeps
  the last of a repeated parameter and returns error bodies that echo the
  input. Both are unacceptable for a credentialed machine API, so the raw
  query string is parsed by hand and unknown/repeated keys are rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from hermes_cli import kanban_occ_read as occ
from hermes_cli.dashboard_auth.base import TokenPrincipal
from hermes_cli.dashboard_auth.token_auth import (
    RouteVerifierDecision,
    register_token_route,
)

log = logging.getLogger(__name__)

SUMMARY_PATH = "/api/plugins/kanban/occ/v1/summary"
TASKS_PATH = "/api/plugins/kanban/occ/v1/tasks"
TASK_PATH = "/api/plugins/kanban/occ/v1/task"
OCC_ROUTE_PATHS: tuple[str, ...] = (SUMMARY_PATH, TASKS_PATH, TASK_PATH)

TOKEN_ENV_VAR = "HERMES_KANBAN_OCC_READ_TOKEN"

#: Sentinel principal. Identity comparison (``is``) — never truthiness — is
#: what the handlers check, so no other principal can impersonate this route.
KANBAN_OCC_PRINCIPAL = TokenPrincipal(
    principal="kanban-occ-read",
    provider="kanban-occ-read",
    scopes=("kanban:occ:read",),
)

_MIN_TOKEN_LEN = 43
_MAX_TOKEN_LEN = 256
_TOKEN68_EXTRA = frozenset("-._~+/=")

#: Forwarding headers are refused outright. A same-host reverse proxy makes
#: every peer look like loopback, so ignoring these would be theatre.
#:
#: Matched by *prefix* (``x-forwarded-``) as well as by name, because an
#: allowlist of known spellings leaks: ``X-Forwarded-Prefix``, ``X-Forwarded-Uri``,
#: ``X-Forwarded-Ssl``, ``X-Forwarded-Scheme``, ``X-Forwarded-Client-Cert`` and
#: anything a future proxy invents would otherwise pass.
_FORWARDING_HEADER_PREFIX = b"x-forwarded-"
_FORWARDING_HEADERS = frozenset({b"forwarded", b"x-real-ip"})
_COOKIE_HEADERS = frozenset({b"cookie", b"x-hermes-session-token"})

_RATE_WINDOW_SECONDS = 60.0
_RATE_BUDGET = 120
_RATE_BURST = 30
_FAILED_BUDGET = 30
#: One summarized audit line per window once a budget is exhausted, instead of
#: one per rejected attempt (B1: the failure budget must *stop* per-attempt
#: auditing, otherwise a guessing loop controls our write amplification).
_SUPPRESSED_AUDIT_TTL_SECONDS = _RATE_WINDOW_SECONDS


class ConfigError(ValueError):
    """Raised when the read surface is misconfigured. Never carries secrets."""


def _load_occ_config() -> tuple[bool, Optional[str]]:
    """Return ``(enabled, board)`` from ``kanban.occ_read``.

    Unreadable config means *disabled*, not "guess a default": the endpoint's
    safe state is closed.
    """
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception as exc:  # noqa: BLE001 — config must not crash the route
        log.warning("kanban-occ-read: config unavailable (%s)", type(exc).__name__)
        return False, None
    section = ((config.get("kanban") or {}).get("occ_read") or {})
    if not isinstance(section, dict):
        return False, None
    enabled = section.get("enabled") is True
    board = section.get("board")
    if board is not None and not isinstance(board, str):
        return False, None
    return enabled, board


def _normalize_board_for_config(board: str) -> str:
    """Validate an explicitly configured board slug.

    The configured value must already equal its normalised form. Silently
    lowercasing it would mean the operator's config and the board actually read
    are two different strings, which is exactly the ambiguity the packet
    forbids.
    """
    from hermes_cli.kanban_db import _normalize_board_slug

    try:
        normalized = _normalize_board_slug(board)
    except ValueError as exc:
        raise ConfigError("configured board slug is invalid") from exc
    if not normalized or normalized != board:
        raise ConfigError("configured board slug is not normalised")
    return normalized


def _kanban_home():
    """The kanban root, resolved through ``kanban_db`` (HOME override aware)."""
    from hermes_cli.kanban_db import kanban_home

    return kanban_home()


def _board_db_path(board: str):
    return occ.resolve_board_db_path(board, kanban_home=_kanban_home())


def _board_exists(board: str) -> bool:
    """A configured board must already exist; it is never created."""
    try:
        return _board_db_path(board).is_file()
    except occ.OccReadError:
        return False


def _valid_configured_token(value: Optional[str]) -> Optional[str]:
    """Accept a token68-shaped secret of 43–256 ASCII characters.

    Length 43 is the base64 length of 32 bytes, so anything shorter cannot be
    a 256-bit secret. The value is never logged, never echoed, and is rejected
    outright if it equals the dashboard session token (which would let a
    browser credential act as a service credential).
    """
    if not value or not isinstance(value, str):
        return None
    if not (_MIN_TOKEN_LEN <= len(value) <= _MAX_TOKEN_LEN):
        return None
    if not value.isascii():
        return None
    if any(not (char.isalnum() or char in _TOKEN68_EXTRA) for char in value):
        return None
    try:
        from hermes_cli import web_server as _ws

        session_token = getattr(_ws, "_SESSION_TOKEN", None)
        if isinstance(session_token, str) and session_token:
            if hmac.compare_digest(value, session_token):
                return None
    except Exception:  # noqa: BLE001 — no dashboard module (tests): nothing to compare
        pass
    return value


#: Snapshot of the validated credential, taken once at route registration
#: (B8). The runbook and the witness both promise "read once at process start,
#: restart to rotate": reading ``os.environ`` per request would make a live
#: ``os.environ`` rewrite (by any code in the process) a hot-reload path the
#: documentation says does not exist. ``_UNSET`` distinguishes "not snapshotted
#: yet" from "snapshotted and absent".
_UNSET: object = object()
_configured_token: object = _UNSET
_TOKEN_LOCK = threading.Lock()


def _snapshot_token() -> Optional[str]:
    """Read, validate and freeze the configured token. Called at registration."""
    return _valid_configured_token(os.environ.get(TOKEN_ENV_VAR))


def _load_token_once() -> Optional[str]:
    """The frozen credential; snapshotted on first use if registration missed it."""
    global _configured_token
    if _configured_token is _UNSET:
        with _TOKEN_LOCK:
            if _configured_token is _UNSET:
                _configured_token = _snapshot_token()
    return _configured_token if isinstance(_configured_token, str) else None


def _reset_token_snapshot() -> None:
    """Test-only: forget the frozen credential so the next call re-reads it."""
    global _configured_token
    with _TOKEN_LOCK:
        _configured_token = _UNSET


class _Budget:
    """Token-bucket budget shared by all authorized OCC read requests.

    One bucket for successes (120/min, burst 30) and one for failures. Keeping
    them separate stops a credential-guessing loop from exhausting the budget a
    legitimate caller needs, while still bounding the write amplification of
    the audit log.

    A failure is debited **only** on a wrong credential (B1): checking a
    well-formed request against the bucket up front, then releasing the debit
    when the comparison succeeds, means ~30 wrong tokens a minute can no longer
    lock the legitimate caller out of the 120/min ceiling it was promised.
    """

    def __init__(self, *, per_minute: int, burst: int) -> None:
        self._rate = per_minute / _RATE_WINDOW_SECONDS
        self._burst = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

    def take(self) -> Optional[float]:
        """Consume one unit. Returns ``None`` on success, else retry-after."""
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return None
            return max(1.0, (1.0 - self._tokens) / self._rate)

    def retry_after(self) -> float:
        """Seconds until this bucket can serve one more request."""
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= 1.0:
                return 0.0
            return max(1.0, (1.0 - self._tokens) / self._rate)


_AuthorizedBudget = _Budget(per_minute=_RATE_BUDGET, burst=_RATE_BURST)
_FailedBudget = _Budget(per_minute=_FAILED_BUDGET, burst=_FAILED_BUDGET)


class _AuditSuppressor:
    """Emit at most one audit record per reason per window.

    Once either budget is exhausted the rejections are no longer individually
    interesting — they are one event, not thousands of log lines. Keyed by
    ``(reason, peer)`` so a flood from one peer cannot hide another peer's
    first failure, and so the very first exhaustion is always recorded.
    """

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._last: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self.suppressed: dict[tuple[str, str], int] = {}

    def allow(self, reason: str, peer: str) -> bool:
        key = (reason, peer)
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and (now - last) < self._ttl:
                self.suppressed[key] = self.suppressed.get(key, 0) + 1
                return False
            self._last[key] = now
            self.suppressed.pop(key, None)
            return True

    def reset(self) -> None:
        with self._lock:
            self._last.clear()
            self.suppressed.clear()


_ExhaustedAudit = _AuditSuppressor(ttl=_SUPPRESSED_AUDIT_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Peer and header checks
# ---------------------------------------------------------------------------

#: A5's allowlist, spelled out. Deliberately *not* ``ipaddress.is_private``:
#: that property also accepts 2001::/23 (which contains the globally reachable
#: Teredo prefix 2001::/32), 2001:db8::/32 and the IPv4 documentation/reserved
#: ranges 192.0.0.0/24, 192.0.2.0/24, 198.18.0.0/15, 198.51.100.0/24 and
#: 203.0.113.0/24. An allowlist cannot leak a range nobody thought of.
_PRIVATE_PEER_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::1/128",
        "fc00::/7",
    )
)


def _peer_address(scope: dict) -> Optional[ipaddress._BaseAddress]:
    """The transport peer as an address, or ``None``.

    ``scope["client"]`` only — never a header. A caller-supplied forwarding
    header is refused separately (B7); trusting one here would make the audit
    ``ip`` whatever the attacker typed.
    """
    client = scope.get("client")
    if not client:
        return None
    host = client[0] if isinstance(client, (tuple, list)) and client else client
    if not isinstance(host, str) or not host:
        return None
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr


def _peer_is_private(scope: dict) -> bool:
    """Loopback, RFC1918 and ULA only — by explicit network membership."""
    addr = _peer_address(scope)
    if addr is None:
        return False
    return any(addr in network for network in _PRIVATE_PEER_NETWORKS)


def _peer_text(scope: dict) -> str:
    """Peer address for the audit record (never a header value)."""
    addr = _peer_address(scope)
    return str(addr) if addr is not None else ""


def _header_values(scope: dict, name: bytes) -> list[bytes]:
    """All raw values for ``name`` in order.

    Counted from ``scope["headers"]`` rather than ``headers.get`` so a repeated
    header is visible: an application that folds duplicates could be tricked
    into accepting the second value. Compared case-insensitively because ASGI
    *requires* lowercase names but a non-conforming server must not be able to
    slip a duplicate past this check.
    """
    target = name.lower()
    return [
        value for key, value in scope.get("headers", []) if key.lower() == target
    ]


def _presented_header_names(scope: dict) -> set[bytes]:
    return {key.lower() for key, _value in scope.get("headers", [])}


def _forwarding_header_present(scope: dict) -> bool:
    """True for ``Forwarded``/``X-Real-IP`` and *any* ``X-Forwarded-*`` name."""
    for name in _presented_header_names(scope):
        if name in _FORWARDING_HEADERS or name.startswith(_FORWARDING_HEADER_PREFIX):
            return True
    return False


def _query_pairs(scope: dict) -> list[tuple[str, str]]:
    """Parse the raw query string, preserving duplicates and order."""
    raw = scope.get("query_string") or b""
    if not raw:
        return []
    pairs: list[tuple[str, str]] = []
    for chunk in raw.split(b"&"):
        if not chunk:
            continue
        key, _, value = chunk.partition(b"=")
        pairs.append(
            (
                unquote_to_bytes(key).decode("utf-8", errors="strict"),
                unquote_to_bytes(value).decode("utf-8", errors="strict"),
            )
        )
    return pairs


#: Reasons refused *before* the credential comparison. These are cheap local
#: checks (config, peer, headers, query, syntax), so an unauthenticated caller
#: can trigger them in a loop without ever presenting a token. Each one is
#: audited, but only once per window per reason: A7 requires the failure budget
#: to stop per-attempt auditing, and a pre-comparison reason never debits the
#: credential budget (B1 round 3 keeps the legitimate caller unthrottled).
_AUDIT_WINDOW_SECONDS: float = 60.0


class _WindowedAudit:
    """At most one audit record per ``reason`` per window.

    Suppression is *per reason* rather than global so a single noisy source
    cannot hide a different attack class from the audit trail. The first record
    of each window is always written, so the trail never goes silent.
    """

    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_write(self, reason: str) -> bool:
        """Record ``reason`` and return whether this occurrence writes a line."""
        with self._lock:
            now = time.monotonic()
            previous = self._last.get(reason)
            if previous is not None and (now - previous) < self._window:
                return False
            self._last[reason] = now
            return True

    def reset(self) -> None:
        """Test-only: forget every window so the next occurrence writes."""
        with self._lock:
            self._last.clear()


_PreComparisonAudit = _WindowedAudit(_AUDIT_WINDOW_SECONDS)


def _reject_audit(reason: str, scope: dict) -> None:
    """Write one peer-attributed audit record for a rejection.

    ``ip`` comes from the transport peer only (B2). The route-scoped path never
    trusts ``X-Forwarded-For``: those requests are refused outright, so deriving
    the IP from the header would record whatever the attacker wrote.

    Pre-comparison reasons are windowed: an unauthenticated loop against
    ``peer_not_private`` or a missing ``Authorization`` header would otherwise
    write one record per request forever (B1).
    """
    from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log

    if not _PreComparisonAudit.should_write(reason):
        return

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        provider="kanban-occ-read",
        reason=reason,
        path=scope.get("path", TASK_PATH),
        ip=_peer_text(scope),
    )


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "unauthenticated"}, status_code=401)


#: The seam's decision type. Defined once in the seam so the middleware can
#: ``isinstance``-check without importing this module (which imports the seam).
VerifierDecision = RouteVerifierDecision


_ACCEPT = VerifierDecision(principal=KANBAN_OCC_PRINCIPAL)


def _throttled(retry_after: float, *, scope: dict, reason: str) -> VerifierDecision:
    """429 for an exhausted budget, audit-suppressed to one record per window."""
    peer = _peer_text(scope)
    if _ExhaustedAudit.allow(reason, peer):
        _reject_audit(reason, scope)
    return VerifierDecision(
        status_code=429,
        error="rate_limited",
        retry_after=max(1, int(math.ceil(retry_after))),
    )


def _verify_occ_request(request: Request) -> VerifierDecision:
    """Route verifier for the three OCC read paths.

    Returns an accepted decision carrying :data:`KANBAN_OCC_PRINCIPAL`, or a
    rejection decision. Every credential failure maps to the same 401 so a
    caller cannot distinguish "disabled" from "wrong token" from "wrong peer"
    by response; only throttling (429) and an unavailable auth backend (503,
    raised by the seam) are distinct, and neither reveals credential validity.

    Order is deliberate and cheap-first where it is safe: config, peer and
    header checks are local and cost nothing, so they run before the token
    comparison. The *failure* budget is checked before the comparison but only
    debited when the comparison fails (B1) — a legitimate caller is never
    charged for someone else's guessing.
    """
    scope = request.scope

    enabled, board = _load_occ_config()
    if not enabled or not board:
        _reject_audit("disabled_or_unconfigured", scope)
        return VerifierDecision()

    configured = _load_token_once()
    if configured is None:
        _reject_audit("token_unconfigured", scope)
        return VerifierDecision()

    if not _peer_is_private(scope):
        _reject_audit("peer_not_private", scope)
        return VerifierDecision()

    if _forwarding_header_present(scope):
        _reject_audit("forwarding_header_present", scope)
        return VerifierDecision()

    if _COOKIE_HEADERS & _presented_header_names(scope):
        _reject_audit("cookie_or_session_header_present", scope)
        return VerifierDecision()

    try:
        pairs = _query_pairs(scope)
    except UnicodeDecodeError:
        # A non-UTF-8 query string is a malformed request, not a credential
        # failure: answering 503 would tell an unauthenticated caller something
        # about the backend, and a warning per request is log spam.
        _reject_audit("query_not_utf8", scope)
        return VerifierDecision()

    if any(key == "token" for key, _value in pairs):
        _reject_audit("query_credential_present", scope)
        return VerifierDecision()

    auth_values = _header_values(scope, b"authorization")
    if len(auth_values) != 1:
        _reject_audit("authorization_header_count", scope)
        return VerifierDecision()
    aauth = auth_values[0]
    if not aauth.startswith(b"Bearer "):
        _reject_audit("authorization_scheme", scope)
        return VerifierDecision()
    presented = aauth[len(b"Bearer "):]
    if not presented or any(byte <= 0x20 or byte >= 0x7F for byte in presented):
        _reject_audit("authorization_token_syntax", scope)
        return VerifierDecision()

    # Check the failure budget *before* doing the comparison work, but debit it
    # only on a mismatch. If it is already exhausted we still allow the request
    # through to the comparison: a correct credential must never be refused
    # because another caller is guessing, and the cost of letting a *wrong*
    # attempt through when the bucket is empty is bounded by the authorized
    # budget below plus the summarized audit (one record per window).
    exhausted_retry = _FailedBudget.retry_after()

    presented_ok = hmac.compare_digest(
        hashlib.sha256(presented).digest(),
        hashlib.sha256(configured.encode("ascii")).digest(),
    )
    if not presented_ok:
        if exhausted_retry > 0:
            return _throttled(exhausted_retry, scope=scope, reason="failure_budget_exhausted")
        _FailedBudget.take()
        _reject_audit("credential_mismatch", scope)
        return VerifierDecision()

    authorized_retry = _AuthorizedBudget.take()
    if authorized_retry is not None:
        return _throttled(authorized_retry, scope=scope, reason="authorized_budget_exhausted")

    return _ACCEPT


def _raw_path_matches(request: Request, path: str) -> bool:
    return (
        request.scope.get("root_path", "") == ""
        and request.scope.get("raw_path") == path.encode("ascii")
    )


def _handler_guard(request: Request, path: str) -> Optional[JSONResponse]:
    """Handler-level identity check.

    The middleware already authenticated. This second check is what stops a
    cookie session that reaches the handler through an unregistered alias path:
    the principal will not be *the* OCC sentinel object, and the raw path will
    not match this handler's own static path.
    """
    principal = getattr(request.state, "token_principal", None)
    if principal is not KANBAN_OCC_PRINCIPAL:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if not _raw_path_matches(request, path):
        return JSONResponse({"error": "not_found"}, status_code=404)
    return None


# ---------------------------------------------------------------------------
# Query parameter parsing
# ---------------------------------------------------------------------------

_ALLOWED_PARAMS = frozenset(
    {"q", "status", "status_class", "assignee", "tenant", "exclude_tenant",
     "order", "limit", "cursor", "task_id"}
)


def _collect_params(request: Request, *, allow_task_id: bool) -> dict[str, str]:
    """Parse query parameters, rejecting unknowns and repeats."""
    try:
        pairs = _query_pairs(request.scope)
    except UnicodeDecodeError as exc:
        raise occ.InvalidRequest("query string is not valid UTF-8") from exc
    params: dict[str, str] = {}
    for key, value in pairs:
        if key not in _ALLOWED_PARAMS:
            raise occ.InvalidRequest("unknown query parameter")
        if key == "task_id" and not allow_task_id:
            raise occ.InvalidRequest("unknown query parameter")
        if key in params:
            raise occ.InvalidRequest("repeated query parameter")
        if len(value) > 512:
            raise occ.InvalidRequest("query parameter too long")
        params[key] = value
    return params


def _filters_from_params(params: dict[str, str]) -> occ.Filters:
    status = occ.validate_status_filter(params.get("status") or None)
    status_class = occ.validate_status_class_filter(params.get("status_class") or None)
    if status is not None and status_class is not None:
        raise occ.InvalidRequest("status and status_class are mutually exclusive")
    return occ.Filters(
        q=occ.normalize_q(params.get("q")),
        assignee=params.get("assignee") or None,
        tenant=params.get("tenant") or None,
        exclude_tenant=params.get("exclude_tenant") or None,
        status=status,
        status_class=status_class,
    )


def _parse_limit(raw: Optional[str]) -> int:
    if raw is None or raw == "":
        return occ.DEFAULT_LIMIT
    try:
        limit = int(raw, 10)
    except ValueError as exc:
        raise occ.InvalidRequest("limit must be an integer") from exc
    if limit < 1 or limit > occ.MAX_LIMIT:
        raise occ.InvalidRequest(f"limit must be 1..{occ.MAX_LIMIT}")
    return limit


def _parse_order(raw: Optional[str], *, filters: occ.Filters) -> str:
    order = raw or ("completed_desc" if filters.status == "done" else "updated_desc")
    return occ.validate_order(order, status=filters.status, status_class=filters.status_class)


def _parse_task_id(raw: Optional[str]) -> str:
    """Task ids are exactly ``t_`` + 8 lowercase hex digits."""
    if not raw:
        raise occ.InvalidTaskId("task_id is required")
    if len(raw) != 10 or not raw.startswith("t_"):
        raise occ.InvalidTaskId("task_id is malformed")
    if not all(char in "0123456789abcdef" for char in raw[2:]):
        raise occ.InvalidTaskId("task_id is malformed")
    return raw


# ---------------------------------------------------------------------------
# Response plumbing
# ---------------------------------------------------------------------------


def _json_response(payload: Any) -> Response:
    try:
        body = occ.enforce_response_size(payload)
    except occ.ResponseTooLarge:
        return _error_response(occ.ResponseTooLarge())
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def _error_response(exc: occ.OccReadError) -> JSONResponse:
    """Error bodies carry a code and nothing else.

    No exception message, no SQL, no filesystem path, no caller input echoed —
    all of which have leaked from naive handlers before.
    """
    return JSONResponse(
        {"error": exc.code},
        status_code=exc.status_code,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def _run_occ_read(operation: Callable[[Any, int], dict[str, Any]], board: str) -> Response:
    """Open one read transaction, run ``operation``, always roll back.

    Handlers are sync ``def`` so FastAPI runs them in the threadpool and a busy
    board cannot stall the dashboard event loop.
    """
    from hermes_cli.kanban_db import DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS

    conn = None
    try:
        conn = occ.open_board(board, kanban_home=_kanban_home())
        now = occ._begin(conn)
        payload = operation(conn, now)
        return _json_response(payload)
    except occ.OccReadError as exc:
        return _error_response(exc)
    except Exception as exc:  # noqa: BLE001 — never leak internals to the caller
        log.warning("kanban-occ-read: read failed (%s)", type(exc).__name__)
        return JSONResponse(
            {"error": "internal_error"},
            status_code=503,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    finally:
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            conn.close()


router = APIRouter()


@router.get("/occ/v1/summary")
def occ_summary(request: Request) -> Response:
    guard = _handler_guard(request, SUMMARY_PATH)
    if guard is not None:
        return guard
    try:
        params = _collect_params(request, allow_task_id=False)
        filters = _filters_from_params(params)
        _parse_order(params.get("order") or "updated_desc", filters=filters)
    except occ.OccReadError as exc:
        return _error_response(exc)
    board = _current_board_or_error()
    if isinstance(board, Response):
        return board

    def op(conn, now: int) -> dict[str, Any]:
        from hermes_cli.kanban_db import DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS

        return occ.read_summary(
            conn,
            board=board,
            filters=filters,
            claim_stale_secs=DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS,
            now=now,
        )

    return _run_occ_read(op, board)


@router.get("/occ/v1/tasks")
def occ_tasks(request: Request) -> Response:
    guard = _handler_guard(request, TASKS_PATH)
    if guard is not None:
        return guard
    try:
        params = _collect_params(request, allow_task_id=False)
        filters = _filters_from_params(params)
        limit = _parse_limit(params.get("limit"))
        order = _parse_order(params.get("order"), filters=filters)
        cursor = params.get("cursor") or None
    except occ.OccReadError as exc:
        return _error_response(exc)
    board = _current_board_or_error()
    if isinstance(board, Response):
        return board

    def op(conn, now: int) -> dict[str, Any]:
        from hermes_cli.kanban_db import DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS

        return occ.read_tasks(
            conn,
            board=board,
            filters=filters,
            order=order,
            limit=limit,
            cursor=cursor,
            claim_stale_secs=DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS,
            now=now,
        )

    return _run_occ_read(op, board)


@router.get("/occ/v1/task")
def occ_task(request: Request) -> Response:
    guard = _handler_guard(request, TASK_PATH)
    if guard is not None:
        return guard
    try:
        params = _collect_params(request, allow_task_id=True)
        task_id = _parse_task_id(params.get("task_id"))
    except occ.OccReadError as exc:
        return _error_response(exc)
    board = _current_board_or_error()
    if isinstance(board, Response):
        return board

    def op(conn, now: int) -> dict[str, Any]:
        from hermes_cli.kanban_db import DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS

        return occ.read_task(
            conn,
            board=board,
            task_id=task_id,
            claim_stale_secs=DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS,
            now=now,
        )

    return _run_occ_read(op, board)


def _current_board_or_error() -> Any:
    """Re-read the configured board for this request.

    Re-read per request rather than cached at import: operators disable the
    surface or change the board through config, and caching would serve a stale
    decision for the process lifetime.
    """
    enabled, board = _load_occ_config()
    if not enabled or not board:
        return _error_response(occ.NotFound("not configured"))
    try:
        normalized = _normalize_board_for_config(board)
    except ConfigError:
        log.warning("kanban-occ-read: configured board is not normalised")
        return _error_response(occ.BoardUnavailable("board misconfigured"))
    if not _board_exists(normalized):
        return _error_response(occ.BoardUnavailable("board missing"))
    return normalized


def register_routes() -> None:
    """Register the three static paths with the scoped verifier.

    Called once at plugin import. Registration is idempotent (the seam allows
    an identical re-registration) and refuses any later binding swap.

    The credential is **not** snapshotted here, and that is deliberate. This
    function runs from the kanban plugin's API module, which ``web_server``
    imports from inside ``_mount_plugin_api_routes()`` while ``web_server`` is
    still executing its own module body. Validation compares the token against
    ``web_server._SESSION_TOKEN``, so snapshotting here would re-enter
    ``hermes_cli.web_server`` mid-import and deadlock the process. This was
    measured, not assumed: registration then blocks forever in the import
    machinery (see the "registration must not snapshot" test and Athena round 3).

    The credential is therefore frozen on the **first authenticated-route
    request**, by which time ``web_server`` has finished importing. That is
    still "process start" for the property that matters: the value is read once,
    frozen for the life of the process, and no later ``os.environ`` rewrite can
    swap it — which is exactly what the runbook and the witness promise. A
    missing or invalid token keeps the surface closed with 401.
    """
    for path in OCC_ROUTE_PATHS:
        register_token_route(path, verifier=_verify_occ_request)
