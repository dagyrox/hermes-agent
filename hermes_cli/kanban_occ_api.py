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
import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from hermes_cli import kanban_occ_read as occ
from hermes_cli.dashboard_auth.base import TokenPrincipal
from hermes_cli.dashboard_auth.token_auth import register_token_route

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
_FORWARDING_HEADERS = frozenset(
    {b"forwarded", b"x-real-ip", b"x-forwarded-for", b"x-forwarded-host",
     b"x-forwarded-proto", b"x-forwarded-port", b"x-forwarded-server"}
)
_COOKIE_HEADERS = frozenset({b"cookie", b"x-hermes-session-token"})

_RATE_WINDOW_SECONDS = 60.0
_RATE_BUDGET = 120
_RATE_BURST = 30
_FAILED_BUDGET = 30


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


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class _Budget:
    """Token-bucket budget shared by all authorized OCC read requests.

    One bucket for successes (120/min, burst 30) and one for failures. Keeping
    them separate stops a credential-guessing loop from exhausting the budget a
    legitimate caller needs, while still bounding the write amplification of
    the audit log.
    """

    def __init__(self, *, per_minute: int, burst: int) -> None:
        self._rate = per_minute / _RATE_WINDOW_SECONDS
        self._burst = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> Optional[float]:
        """Consume one unit. Returns ``None`` on success, else retry-after."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._updated
            self._updated = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return None
            return max(1.0, (1.0 - self._tokens) / self._rate)


_AuthorizedBudget = _Budget(per_minute=_RATE_BUDGET, burst=_RATE_BURST)
_FailedBudget = _Budget(per_minute=_FAILED_BUDGET, burst=_FAILED_BUDGET)


# ---------------------------------------------------------------------------
# Peer and header checks
# ---------------------------------------------------------------------------


def _peer_is_private(scope: dict) -> bool:
    """Loopback / RFC1918 / ULA only. Link-local, CGNAT, unspecified refused."""
    client = scope.get("client")
    if not client:
        return False
    host = client[0] if isinstance(client, (tuple, list)) and client else client
    if not isinstance(host, str) or not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if addr.is_loopback or addr.is_private and not addr.is_link_local:
        # is_private also covers link-local and CGNAT in some versions, so
        # exclude those explicitly; ULA (fc00::/7) is already is_private.
        if addr.is_link_local:
            return False
        if isinstance(addr, ipaddress.IPv4Address) and addr in ipaddress.ip_network(
            "100.64.0.0/10"
        ):
            return False
        if addr.is_unspecified or addr.is_multicast or addr.is_reserved:
            return False
        return True
    return False


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


def _reject_audit(reason: str, scope: dict) -> None:
    from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log

    client = scope.get("client")
    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        provider="kanban-occ-read",
        reason=reason,
        path=scope.get("path", TASK_PATH),
        ip=(client[0] if isinstance(client, (tuple, list)) and client else ""),
    )


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "unauthenticated"}, status_code=401)


def _verify_occ_request(request: Request) -> Optional[TokenPrincipal]:
    """Route verifier for the three OCC read paths.

    Returns :data:`KANBAN_OCC_PRINCIPAL` on success, ``None`` on any failure.
    Every rejection maps to the same 401 so a caller cannot distinguish
    "disabled" from "wrong token" from "wrong peer" by response.
    """
    scope = request.scope

    enabled, board = _load_occ_config()
    if not enabled or not board:
        _reject_audit("disabled_or_unconfigured", scope)
        return None

    import os

    configured = _valid_configured_token(os.environ.get(TOKEN_ENV_VAR))
    if configured is None:
        _reject_audit("token_unconfigured", scope)
        return None

    if not _peer_is_private(scope):
        _reject_audit("peer_not_private", scope)
        return None

    headers = scope.get("headers", [])
    presented_headers = {key.lower() for key, _value in headers}
    for banned in (_FORWARDING_HEADERS | _COOKIE_HEADERS) & presented_headers:
        _reject_audit(
            "forwarding_header_present" if banned in _FORWARDING_HEADERS
            else "cookie_or_session_header_present",
            scope,
        )
        return None

    if _query_pairs(scope):
        for key, _value in _query_pairs(scope):
            if key == "token":
                _reject_audit("query_credential_present", scope)
                return None

    auth_values = _header_values(scope, b"authorization")
    if len(auth_values) != 1:
        _reject_audit("authorization_header_count", scope)
        return None
    aauth = auth_values[0]
    if not aauth.startswith(b"Bearer "):
        _reject_audit("authorization_scheme", scope)
        return None
    presented = aauth[len(b"Bearer "):]
    if not presented or any(byte <= 0x20 or byte >= 0x7F for byte in presented):
        _reject_audit("authorization_token_syntax", scope)
        return None

    retry_after = _FailedBudget.take()
    if retry_after is not None:
        _reject_audit("attempt_budget_exhausted", scope)
        return None

    presented_ok = hmac.compare_digest(
        hashlib.sha256(presented).digest(),
        hashlib.sha256(configured.encode("ascii")).digest(),
    )
    if not presented_ok:
        _reject_audit("credential_mismatch", scope)
        return None

    if _AuthorizedBudget.take() is not None:
        _reject_audit("authorized_budget_exhausted", scope)
        return None

    return KANBAN_OCC_PRINCIPAL


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
    """
    for path in OCC_ROUTE_PATHS:
        register_token_route(path, verifier=_verify_occ_request)
