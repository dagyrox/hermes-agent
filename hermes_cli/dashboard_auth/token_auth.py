"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

This is the generic API-token capability (decisions.md Q-C): a reusable seam
that ANY service-to-service / machine-credential provider plugs into, NOT a
drain-specific hook. The drain bearer-secret plugin is merely the first
consumer.

How it fits the existing auth framework:

  * The interactive gate (``gated_auth_middleware``) authenticates a human
    via a session cookie on every non-public route. A service caller has no
    cookie — it presents a bearer token in the ``Authorization`` header on a
    single request. That is what this seam verifies.

  * A route opts in by registering its exact path via
    :func:`register_token_route`. Only registered paths are token-authable;
    everything else is untouched, so this can never accidentally widen the
    auth surface of an existing route.

  * :func:`token_auth_middleware` runs OUTERMOST (installed last in
    ``web_server.py``). For a token route it fully owns the auth decision:
    authenticate via the stacked token providers, attach the verified
    :class:`~hermes_cli.dashboard_auth.base.TokenPrincipal` to
    ``request.state.token_principal`` + set ``request.state.token_authenticated``,
    and pass through; otherwise reject (401 unauthenticated, or 503 when a
    provider's backing store was unreachable). The downstream cookie/session
    gates honour ``token_authenticated`` and skip enforcement, so a
    token-authed service request is never bounced to ``/login``.

  * Fails closed: a token route with no registered token provider, no token,
    or an unrecognised token gets 401 — never an open pass-through.

Provider stacking mirrors ``verify_session``: each ``supports_token`` provider
is consulted in registration order until one returns a principal. A provider
that doesn't recognise the token returns ``None`` and the seam moves on; a
provider whose backing store is unreachable raises ``ProviderError``, which the
seam remembers and surfaces as 503 only if NO provider accepts the token.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal

_log = logging.getLogger(__name__)

# Exact paths that accept non-interactive bearer-token auth. A route registers
# itself here at import/startup; the seam only acts on registered paths.
_token_routes: set[str] = set()
# Route-local verifiers. A path registered with its own verifier NEVER consults
# the global provider stack: that keeps a route-scoped credential from
# authenticating anywhere else (e.g. the gateway drain route). A verifier may
# return ``(principal, unreachable)`` like :func:`authenticate_token`, or just
# a bare ``principal | None``.
TokenVerifier = Callable[[Request], Any]
_route_verifiers: dict[str, TokenVerifier] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class RouteVerifierDecision:
    """Optional rich rejection for a route verifier.

    A verifier may return one of these instead of ``principal | None`` when the
    rejection needs a status other than 401 — throttling is the motivating case
    (429 + ``Retry-After``). The dataclass lives here, not in the consumer, so
    any future route-scoped provider can use it without importing that module.
    ``principal is None`` means "rejected".
    """

    principal: Optional[TokenPrincipal] = None
    status_code: int = 401
    error: str = "unauthenticated"
    retry_after: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.principal is not None

# Canonical token-route paths are narrow by construction: ASCII, absolute, no
# percent-encoding, query/fragment, empty or dot segments, or trailing slash.
# Anything else could alias another path once a proxy or router normalises it.
_UNSAFE_PATH_CHARS = frozenset("%?#\\")


def _is_canonical_route_path(path: object) -> bool:
    if not isinstance(path, str) or not path.isascii() or not path.startswith("/"):
        return False
    if path.endswith("/") or "//" in path:
        return False
    if _UNSAFE_PATH_CHARS.intersection(path):
        return False
    if any(ord(char) <= 0x20 or ord(char) == 0x7F for char in path):
        return False
    return not any(segment in {".", ".."} for segment in path.split("/"))


def register_token_route(path: str, *, verifier: Optional[TokenVerifier] = None) -> None:
    """Mark ``path`` (exact match) as token-authable.

    Idempotent. Call at module import / app setup so the seam knows which
    routes to guard. Registering a route does NOT make it public — it makes
    it authenticate by token instead of by session cookie.

    When ``verifier`` is supplied the route is authenticated *only* by that
    verifier; the global provider stack is never consulted for it. The
    verifier binding is immutable for the lifetime of the registration:
    re-registering the same path with a different verifier (including
    switching between ``None`` and a verifier) raises :class:`ValueError`
    so a later plugin cannot silently take over an existing route.
    """
    if not _is_canonical_route_path(path):
        raise ValueError(
            "token route must be a canonical exact ASCII path "
            "(absolute, no percent-encoding, query, fragment, empty/dot "
            "segments or trailing slash)"
        )
    with _lock:
        if path in _route_verifiers:
            existing = _route_verifiers[path]
            if existing is not verifier:
                raise ValueError(
                    f"token route {path!r} is already registered with a "
                    "different verifier"
                )
        elif path in _token_routes and verifier is not None:
            raise ValueError(
                f"token route {path!r} is already registered without a verifier"
            )
        _token_routes.add(path)
        if verifier is not None:
            _route_verifiers[path] = verifier


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable (exact match)."""
    with _lock:
        return path in _token_routes


def route_verifier(path: str) -> Optional[TokenVerifier]:
    """Return the verifier bound to ``path``, or ``None``.

    Shared by the middleware and by handler-level identity guards so the two
    can never disagree about which paths are route-scoped.
    """
    with _lock:
        return _route_verifiers.get(path)


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes and their verifiers."""
    with _lock:
        _token_routes.clear()
        _route_verifiers.clear()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def extract_bearer_token(request: Request) -> str:
    """Return the bearer token from the ``Authorization`` header, or "".

    Accepts ``<scheme> <token>`` where scheme is "bearer" (case-insensitive).
    Returns an empty string for a missing/malformed header or a non-bearer
    scheme — the caller treats "" as "no token presented".
    """
    auth = request.headers.get("authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def authenticate_token(
    request: Request,
) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token.

    Returns ``(principal, unreachable_provider_name)``:
      * ``(TokenPrincipal, None)`` — a provider recognised and accepted the token.
      * ``(None, None)`` — no token, or no provider recognised it (reject 401).
      * ``(None, name)`` — no provider accepted it AND at least one provider's
        backing store was unreachable (the caller surfaces 503, not 401, so a
        transient outage doesn't read as "bad credentials").

    Never raises: a provider ``ProviderError`` is caught and remembered.
    """
    token = extract_bearer_token(request)
    if not token:
        return None, None
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: token provider %r unreachable during verify: %s",
                provider.name, e,
            )
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:  # noqa: BLE001 — a buggy provider must not 500 the gate
            _log.warning(
                "dashboard-auth: token provider %r raised during verify: %s",
                provider.name, e,
            )
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


async def _route_scoped_token_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    path: str,
    verifier: TokenVerifier,
) -> Response:
    """Authorize one request on a route that carries its own verifier.

    Order matters and is deliberate:

    1. Raw-path identity first. ``request.url.path`` is percent-decoded by the
       ASGI server, so ``/x%2Fy`` and ``/x/y`` can collapse to the same string.
       A route-scoped credential must only ever unlock the exact bytes the
       caller sent, and a non-empty ``root_path`` (a mount prefix) means the
       app is not serving this path directly.
    2. The route verifier, and *only* it. No global provider is consulted, so a
       scoped read credential can never authenticate on another token route.
    3. Method check *after* authentication: an unauthenticated caller learns
       nothing about the route's method set, and an authorized non-GET is
       refused here rather than in a handler that may mutate state.

    HEAD/OPTIONS are included in the refusal: Starlette would otherwise answer
    them without ever invoking the handler.

    **Audit ownership.** A route that binds its own verifier owns its audit:
    the verifier knows *why* it refused and can attribute the peer correctly,
    and only it can tell a wrong credential from an exhausted budget. This
    seam therefore writes **no** record for a verifier-route rejection (B2) —
    the old second ``route_credential_rejected`` line both duplicated every
    failure and, through ``_client_ip``, took its IP from ``X-Forwarded-For``.
    Legacy non-verifier routes keep their existing seam auditing unchanged.
    """
    if request.scope.get("root_path", "") != "":
        return _unauthorized(request, path, reason="non_empty_root_path", audit=False)
    if request.scope.get("raw_path") != path.encode("ascii"):
        return _unauthorized(request, path, reason="raw_path_mismatch", audit=False)

    try:
        result = verifier(request)
        # Accept the seam's ``(principal, unreachable)`` shape, a bare
        # ``principal | None``, or a rich decision object carrying an explicit
        # status (429 + Retry-After for throttling). Tolerating all three keeps
        # a simple verifier simple without forcing every rejection through 401.
        if isinstance(result, tuple) and len(result) == 2:
            principal = result[0]
            rejection: Optional[Response] = None
        elif isinstance(result, RouteVerifierDecision):
            principal = result.principal if result.ok else None
            rejection = None if result.ok else _decision_response(result)
        else:
            principal = result
            rejection = None
    except Exception as exc:  # noqa: BLE001 — auth boundary: fail closed, no text
        _log.warning(
            "dashboard-auth: route verifier for %s raised %s",
            path,
            type(exc).__name__,
        )
        return JSONResponse({"error": "auth_unavailable"}, status_code=503)

    if principal is None:
        if rejection is not None:
            return rejection
        return _unauthorized(request, path, reason="route_credential_rejected", audit=False)

    if request.method != "GET":
        return JSONResponse(
            {"error": "method_not_allowed"},
            status_code=405,
            headers={"Allow": "GET"},
        )

    request.state.token_principal = principal
    request.state.token_authenticated = True
    return await call_next(request)


def _decision_response(decision: RouteVerifierDecision) -> JSONResponse:
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if decision.retry_after is not None:
        headers["Retry-After"] = str(decision.retry_after)
    return JSONResponse(
        {"error": decision.error},
        status_code=decision.status_code,
        headers=headers,
    )


def _unauthorized(request: Request, path: str, *, reason: str, audit: bool = True) -> JSONResponse:
    """Uniform 401 for route-scoped credentials (reason is audit-only).

    ``audit=False`` is used on paths owned by a route verifier, which has
    already written its own single, peer-correct record.
    """
    if audit:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE,
            provider="route-scoped",
            reason=reason,
            path=path,
            ip=_client_ip(request),
        )
    return JSONResponse({"error": "unauthenticated"}, status_code=401)


async def token_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Outermost auth seam for token-authable routes.

    No-op pass-through for any path not registered via
    :func:`register_token_route`. For a registered path, token auth is the
    only accepted scheme:

      * valid token  → attach principal + ``token_authenticated`` flag, pass through.
      * unreachable  → 503 (provider backing store down; not "bad credentials").
      * otherwise    → 401 unauthenticated.

    Runs before the cookie/session gates (installed last in ``web_server.py``).
    The cookie gates honour ``request.state.token_authenticated`` and skip
    enforcement, so a token-authed request is never redirected to ``/login``.
    """
    path = request.url.path
    if not is_token_route(path):
        return await call_next(request)

    verifier = route_verifier(path)
    if verifier is not None:
        return await _route_scoped_token_request(request, call_next, path, verifier)

    principal, unreachable = authenticate_token(request)
    if principal is not None:
        request.state.token_principal = principal
        request.state.token_authenticated = True
        return await call_next(request)

    if unreachable:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE,
            provider=unreachable,
            reason="provider_unreachable",
            path=path,
            ip=_client_ip(request),
        )
        return JSONResponse(
            {"detail": f"Auth provider {unreachable!r} unreachable"},
            status_code=503,
        )

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        reason="no_provider_recognises_token",
        path=path,
        ip=_client_ip(request),
    )
    return JSONResponse(
        {"error": "unauthenticated", "detail": "Unauthorized"},
        status_code=401,
    )
