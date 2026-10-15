from fastapi.testclient import TestClient

from app.api.main import create_app


def test_health(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").status_code == 200


def test_demo_direct_generation_is_structured_error(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer test-reader-token"},
            json={"prompt": "hello"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_disabled"
