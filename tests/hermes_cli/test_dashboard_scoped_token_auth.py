"""Adversarial tests for route-scoped token auth (OCC #263, Athena A-H).

The risk this suite exists to prevent: the generic token seam stacks *global*
providers, so a credential registered globally for a read-only route would also
authenticate at every other registered token route — including the gateway
drain route, whose handler checks no scope and changes state.

Every test drives the real middleware with a real ASGI ``Request`` and a real
Starlette app where routing behaviour matters, so a change in Starlette's
path handling shows up here rather than in production.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from hermes_cli.dashboard_auth import TokenPrincipal, clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth


class _TokenProvider:
    """A global provider that accepts exactly one secret (real ABC subclass)."""

    name = "global-token"
    display_name = "Global"
    supports_token = True

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def start_login(self, *, redirect_uri):
        raise NotImplementedError

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        raise NotImplementedError

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        raise NotImplementedError

    def revoke_session(self, *, refresh_token):
        return None

    def verify_token(self, *, token: str):
        if token == self._secret:
            return TokenPrincipal("global", "global-token", ())
        return None


@pytest.fixture(autouse=True)
def _clean_seam():
    clear_providers()
    token_auth.clear_token_routes()
    yield
    clear_providers()
    token_auth.clear_token_routes()


def _run(coro):
    return asyncio.run(coro)


def _request(path: str, *, headers=None, method="GET", root_path="", raw_path=None,
             client=("127.0.0.1", 5000)):
    # ASGI requires lowercase header names in scope["headers"]; Starlette's
    # Headers lookup is case-sensitive on the raw scope even though HTTP is not.
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": raw_path if raw_path is not None else path.encode("ascii"),
        "root_path": root_path,
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or [])
        ],
        "client": client,
        "query_string": b"",
    }
    return Request(scope)


async def _ok(request):
    return JSONResponse({"ok": True}, status_code=200)


def _accepted_verifier():
    def verifier(request):
        return TokenPrincipal("occ", "kanban-occ-read", ("kanban:occ:read",)), None

    return verifier


# ---------------------------------------------------------------------------
# Registration semantics
# ---------------------------------------------------------------------------


def test_repeated_identical_registration_is_allowed():
    verifier = _accepted_verifier()
    token_auth.register_token_route("/machine", verifier=verifier)
    token_auth.register_token_route("/machine", verifier=verifier)
    assert token_auth.is_token_route("/machine")


def test_rebinding_a_verifier_raises():
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    with pytest.raises(ValueError):
        token_auth.register_token_route("/machine", verifier=_accepted_verifier())


def test_switching_between_none_and_verifier_raises_both_ways():
    token_auth.register_token_route("/plain")
    with pytest.raises(ValueError):
        token_auth.register_token_route("/plain", verifier=_accepted_verifier())

    token_auth.register_token_route("/scoped", verifier=_accepted_verifier())
    with pytest.raises(ValueError):
        token_auth.register_token_route("/scoped")


@pytest.mark.parametrize(
    "path",
    [
        "relative/path",
        "/trailing/",
        "/double//slash",
        "/dot/./segment",
        "/dot/../segment",
        "/percent%2Fencoded",
        "/query?x=1",
        "/fragment#x",
        "/space here",
        "/control\x00char",
        "/non-ascii-é",
        "",
    ],
)
def test_router_rejects_non_canonical_paths(path):
    with pytest.raises(ValueError):
        token_auth.register_token_route(path)


def test_clear_token_routes_clears_verifiers():
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    token_auth.clear_token_routes()
    assert token_auth.route_verifier("/machine") is None
    # A path may be re-registered with a different verifier after clearing.
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    assert token_auth.is_token_route("/machine")


# ---------------------------------------------------------------------------
# Credential isolation (Athena A1/A2 + H)
# ---------------------------------------------------------------------------


def test_route_verifier_never_consults_the_global_provider_stack(monkeypatch):
    def _boom(request):
        raise AssertionError("global provider stack must not be consulted")

    monkeypatch.setattr(token_auth, "authenticate_token", _boom)
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    resp = _run(token_auth.token_auth_middleware(_request("/machine"), _ok))
    assert resp.status_code == 200


def test_occ_credential_does_not_authenticate_on_the_drain_route():
    """A scoped read credential must not unlock a state-changing route.

    B9 round 3: this used to assert only the seam's 401 against a stub handler.
    It now drives the real drain handler and asserts no drain marker file was
    written — the state change the credential must never be able to cause. It
    also uses a token shaped like a real OCC credential rather than a stub.
    """
    import os
    import tempfile
    from pathlib import Path

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from gateway import drain_control
    from hermes_cli import web_server

    occ_token = "O" * 43
    token_auth.register_token_route("/api/gateway/drain")
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())

    # The global stack has no provider that accepts this token, exactly as in
    # production: the OCC credential is never registered globally.
    resp = _run(
        token_auth.token_auth_middleware(
            _request("/api/gateway/drain", headers=[("Authorization", f"Bearer {occ_token}")]),
            _ok,
        )
    )
    assert resp.status_code == 401

    # End to end: mount the real handler behind the real seam and prove the
    # marker file is never created by an unauthenticated POST.
    with tempfile.TemporaryDirectory() as home_dir:
        old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = home_dir
        try:
            marker = drain_control.drain_request_path()
            drain_app = FastAPI()
            drain_app.middleware("http")(token_auth.token_auth_middleware)
            drain_app.add_api_route(
                "/api/gateway/drain", web_server.gateway_drain, methods=["POST"]
            )
            with TestClient(drain_app, client=("127.0.0.1", 40000)) as client:
                post = client.post(
                    "/api/gateway/drain",
                    json={"action": "drain"},
                    headers={"Authorization": f"Bearer {occ_token}"},
                )
            assert post.status_code == 401
            assert not Path(marker).exists(), f"drain marker was written: {marker}"
        finally:
            if old_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old_home


def test_legacy_unverifiered_route_behaviour_is_unchanged():
    register_provider(_TokenProvider("legacy-secret"))
    token_auth.register_token_route("/api/gateway/drain")
    ok = _run(
        token_auth.token_auth_middleware(
            _request("/api/gateway/drain", headers=[("Authorization", "Bearer legacy-secret")]),
            _ok,
        )
    )
    assert ok.status_code == 200
    bad = _run(token_auth.token_auth_middleware(_request("/api/gateway/drain"), _ok))
    assert bad.status_code == 401


# ---------------------------------------------------------------------------
# Raw path / root path identity (Athena A2a + H)
# ---------------------------------------------------------------------------


def test_missing_raw_path_is_rejected():
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    request = _request("/machine")
    request.scope.pop("raw_path")
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401


def test_non_empty_root_path_is_rejected():
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    request = _request("/machine", root_path="/mounted")
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401


@pytest.mark.parametrize(
    "raw",
    [b"/machine/", b"/MACHINE", b"/machine%2F", b"/mac%68ine", b"//machine",
     b"/./machine", b"/mach ine"],
)
def test_raw_path_aliases_do_not_authenticate(raw):
    """``url.path`` may decode to the registered path; ``raw_path`` may not."""
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    request = _request("/machine", raw_path=raw)
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401


@pytest.fixture()
def seam_audit(monkeypatch):
    """Capture records written by the seam itself (R3-1)."""
    records: list[dict] = []

    def _capture(event, **fields):
        records.append({"event": getattr(event, "value", str(event)), **fields})

    monkeypatch.setattr(token_auth, "audit_log", _capture)
    return records


def test_non_empty_root_path_is_audited_once_from_the_peer(seam_audit):
    """R3-1: an alias probe must not be silent.

    The B2 repair originally passed ``audit=False`` here and dropped this record
    entirely, which made the alias probes the audit exists to catch invisible.
    """
    token_auth.reset_alias_audit_window()
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    request = _request(
        "/machine",
        root_path="/mounted",
        headers=[("X-Forwarded-For", "203.0.113.66")],
        client=("192.168.5.5", 5000),
    )
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401
    assert len(seam_audit) == 1, seam_audit
    record = seam_audit[0]
    assert record["reason"] == "non_empty_root_path"
    # The IP is the transport peer, not the forwarding header.
    assert record["ip"] == "192.168.5.5", record
    assert "203.0.113.66" not in json.dumps(record)


def test_raw_path_alias_is_audited_once_from_the_peer(seam_audit):
    """R3-1: the encoded-path variant is audited exactly once."""
    token_auth.reset_alias_audit_window()
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    request = _request(
        "/machine",
        raw_path=b"/mac%68ine",
        headers=[("X-Forwarded-For", "203.0.113.66")],
        client=("192.168.5.5", 5000),
    )
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401
    assert len(seam_audit) == 1, seam_audit
    assert seam_audit[0]["reason"] == "raw_path_mismatch"
    assert seam_audit[0]["ip"] == "192.168.5.5", seam_audit[0]


def test_alias_probe_audit_is_windowed_r4_1(seam_audit):
    """R4-1: 20 alias probes must write one record, not 20.

    Alias probes are triggerable from any peer by an encoded path, so an
    unwindowed record is a write amplifier — the same defect B1 fixed for the
    verifier's pre-comparison checks.
    """
    token_auth.reset_alias_audit_window()
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    for _ in range(20):
        request = _request("/machine", raw_path=b"/mac%68ine", client=("192.168.5.5", 5000))
        assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401
    assert len(seam_audit) == 1, seam_audit
    assert seam_audit[0]["reason"] == "raw_path_mismatch"


def test_verifier_route_rejection_is_not_double_audited(seam_audit):
    """B2/R3-2: the seam stays silent when the verifier owns the rejection.

    With ``token_auth.audit_log`` now patched, this assertion can actually fail
    if the seam reintroduces its second record.
    """
    def _rejecting_verifier(request):
        return None  # verifier refuses; it would audit in product code

    token_auth.register_token_route("/machine", verifier=_rejecting_verifier)
    request = _request("/machine")
    assert _run(token_auth.token_auth_middleware(request, _ok)).status_code == 401
    assert seam_audit == [], seam_audit


# ---------------------------------------------------------------------------
# Method handling (Athena A2c + H) — real routing, no handler reached
# ---------------------------------------------------------------------------


def _scoped_app():
    token_auth.register_token_route("/machine", verifier=_accepted_verifier())
    app = FastAPI()
    app.middleware("http")(token_auth.token_auth_middleware)
    hits: list[str] = []

    @app.get("/machine")
    async def machine():
        hits.append("GET")
        return {"ok": True}

    @app.post("/machine")
    async def machine_post():
        hits.append("POST")
        return {"ok": True}

    return app, hits


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_non_get_is_refused_before_reaching_the_handler(method):
    app, hits = _scoped_app()
    with TestClient(app) as client:
        resp = client.request(method, "/machine")
    assert resp.status_code == 405
    assert resp.headers["allow"] == "GET"
    assert hits == []


def test_unauthenticated_non_get_is_rejected_without_reaching_the_handler():
    """An unauthenticated non-GET never reaches a handler.

    Starlette answers 405 from the router for an unrouted method on a real app,
    which is standard and discloses only that a path exists (not that it is a
    token route). What must never happen is the handler running, so that is what
    this pins.
    """
    app, hits = _scoped_app()
    with TestClient(app) as client:
        resp = client.post("/machine")
    assert resp.status_code in {401, 405}
    assert hits == []


def test_unauthenticated_non_get_with_route_present_returns_405_from_router():
    """Documented Starlette behaviour: a routed non-GET 405s before middleware."""
    app, hits = _scoped_app()
    with TestClient(app) as client:
        resp = client.post("/machine", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 405
    assert hits == []


def test_authenticated_get_reaches_the_handler():
    app, hits = _scoped_app()
    with TestClient(app) as client:
        resp = client.get("/machine")
    assert resp.status_code == 200
    assert hits == ["GET"]


# ---------------------------------------------------------------------------
# Verifier failure modes (Athena A7 + H)
# ---------------------------------------------------------------------------


def test_verifier_exception_returns_503_without_leaking_text():
    def verifier(request):
        raise RuntimeError("database password is hunter2")

    token_auth.register_token_route("/machine", verifier=verifier)
    resp = _run(token_auth.token_auth_middleware(_request("/machine"), _ok))
    assert resp.status_code == 503
    assert json.loads(resp.body) == {"error": "auth_unavailable"}
    assert b"hunter2" not in resp.body


def test_rejection_body_is_a_uniform_401_code():
    token_auth.register_token_route("/machine", verifier=lambda request: (None, None))
    resp = _run(token_auth.token_auth_middleware(_request("/machine"), _ok))
    assert resp.status_code == 401
    assert json.loads(resp.body) == {"error": "unauthenticated"}
