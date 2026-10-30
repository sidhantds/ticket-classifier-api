from fastapi.testclient import TestClient
from app.main import HIGH_PRIORITY, app, model

client = TestClient(app)


# Happy path tests: correct input gives correct output

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_all_fields():
    response = client.post("/predict", json={"text": "where is my package"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"text", "category", "priority", "confidence"}
    assert 0 <= data["confidence"] <= 1
    assert data["priority"] in {"high", "normal"}


def test_payment_ticket_is_high_priority():
    response = client.post("/predict", json={"text": "my payment failed and money was deducted"})
    data = response.json()
    assert data["category"] == "PAYMENT"
    assert data["priority"] == "high"


def test_cancellation_fee_ticket_is_high_priority():
    response = client.post("/predict", json={"text": "what is the cancellation fee"})
    data = response.json()
    assert data["category"] == "CANCEL"
    assert data["priority"] == "high"


def test_high_priority_categories_exist_in_model():
    assert HIGH_PRIORITY <= set(model.classes_)


def test_batch_returns_one_prediction_per_ticket():
    body = {"tickets": [{"text": "cancel my order"}, {"text": "how do i get an invoice"}]}
    response = client.post("/predict/batch", json=body)
    assert response.status_code == 200
    assert len(response.json()) == 2


# Validation tests: bad input must be rejected with 422

def test_short_text_rejected():
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422


def test_whitespace_text_rejected():
    response = client.post("/predict", json={"text": "      "})
    assert response.status_code == 422


def test_too_long_text_rejected():
    response = client.post("/predict", json={"text": "a" * 2001})
    assert response.status_code == 422


def test_missing_text_field_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_empty_batch_rejected():
    response = client.post("/predict/batch", json={"tickets": []})
    assert response.status_code == 422


def test_batch_over_limit_rejected():
    body = {"tickets": [{"text": "cancel my order"}] * 101}
    response = client.post("/predict/batch", json=body)
    assert response.status_code == 422