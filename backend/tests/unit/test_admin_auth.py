"""Unit tests for the admin API key dependency."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_settings_dep, verify_admin_key
from src.core.config import Settings


def _make_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    app.dependency_overrides[get_settings_dep] = lambda: settings

    @app.get("/protected", dependencies=[Depends(verify_admin_key)])
    def protected():
        return {"ok": True}

    return app


class TestVerifyAdminKey:
    def test_allows_request_when_key_unset(self):
        client = TestClient(_make_app(Settings(admin_api_key=None)))
        response = client.get("/protected")
        assert response.status_code == 200

    def test_rejects_missing_header_when_key_set(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected")
        assert response.status_code == 401

    def test_rejects_wrong_key(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected", headers={"X-Admin-Key": "wrong"})
        assert response.status_code == 401

    def test_accepts_correct_key(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected", headers={"X-Admin-Key": "secret"})
        assert response.status_code == 200
