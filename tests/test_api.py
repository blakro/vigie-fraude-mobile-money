from fastapi.testclient import TestClient

from api.app import create_app
from vigie.engine.engine import FraudEngine


def client():
    return TestClient(create_app(FraudEngine()))


TX = {"txn_id": "DX260830.000001", "timestamp": "2026-08-30T10:00:00", "txn_type": "TRANSFERT",
      "account_id": "C4N7QX2A", "counterparty_id": "CB2HT9KD", "counterparty_type": "USER",
      "amount": 5000, "balance_before": 152300, "device_id": "5e33124b8a19"}


def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_normal_then_account_takeover():
    with client() as c:
        assert c.post("/score", json=TX).json()["decision"] == "ACCEPTER"
        c.post("/events", json={"account_id": "C4N7QX2A", "timestamp": "2026-08-30T10:05:00",
                                "event_type": "REINIT_PIN", "device_id": "9f3c2a71be04"})
        c.post("/events", json={"account_id": "C4N7QX2A", "timestamp": "2026-08-30T10:06:00",
                                "event_type": "CONNEXION_NOUVEL_APPAREIL", "device_id": "9f3c2a71be04"})
        body = c.post("/score", json={**TX, "txn_id": "DX260830.000002", "timestamp": "2026-08-30T10:10:00",
                                      "counterparty_id": "CZ8RKM3T", "amount": 145000, "balance_before": 147250,
                                      "device_id": "9f3c2a71be04"}).json()
        assert body["decision"] == "BLOQUER"
        assert {"reinit_pin", "nouvel_appareil", "vidage_compte"} <= set(body["signals"])
        assert body["reasons"]
        assert c.get("/metrics").json()["operations_scorees"] == 2


def test_invalid_payload_rejected():
    with client() as c:
        assert c.post("/score", json={**TX, "amount": -10}).status_code == 422
        assert c.post("/score", json={**TX, "txn_type": "COFFRE"}).status_code == 422
