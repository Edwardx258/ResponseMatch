from fastapi.testclient import TestClient

from responsematch.api import create_app


def test_health_and_match(deps):
    client = TestClient(create_app(deps))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["volunteers"] == 8 and health["handbook_chunks"] > 0

    resp = client.post(
        "/match",
        json={"text": "Someone fell and is injured, need first aid", "lat": 47.565, "lon": -122.386, "mode": "dual_track"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "dual_track"
    assert body["matches"][0]["volunteer_id"] == "v04" and body["matches"][0]["recommended"]


def test_bad_mode_is_rejected(deps):
    client = TestClient(create_app(deps))
    assert client.post("/match", json={"text": "x", "mode": "nope"}).status_code == 422
