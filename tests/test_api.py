from fastapi.testclient import TestClient

from src.api import app


# Create a test client
client = TestClient(app)


# ============================================================
# HEALTH CHECK
# ============================================================

def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ============================================================
# ACCOUNT ACCESS
# ============================================================

def test_account_access():

    response = client.post(
        "/predict",
        json={
            "text": "I forgot my password and cannot log in"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Account Access"
    assert data["is_known"] is True


# ============================================================
# BILLING
# ============================================================

def test_billing():

    response = client.post(
        "/predict",
        json={
            "text": "My credit card was charged twice"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Billing"
    assert data["is_known"] is True


# ============================================================
# TECHNICAL ISSUE
# ============================================================

def test_technical_issue():

    response = client.post(
        "/predict",
        json={
            "text": "The application keeps crashing"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Technical Issue"
    assert data["is_known"] is True


# ============================================================
# GENERAL INQUIRY
# ============================================================

def test_general_inquiry():

    response = client.post(
        "/predict",
        json={
            "text": "How can I contact customer support?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "General Inquiry"
    assert data["is_known"] is True


# ============================================================
# UNKNOWN MESSAGE
# ============================================================

def test_unknown_message():

    response = client.post(
        "/predict",
        json={
            "text": "hello"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Unknown"
    assert data["is_known"] is False


# ============================================================
# EMPTY MESSAGE
# ============================================================

def test_empty_message():

    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 422


# ============================================================
# MISSING TEXT
# ============================================================

def test_missing_text():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# ============================================================
# MESSAGE TOO SHORT
# ============================================================

def test_message_too_short():

    response = client.post(
        "/predict",
        json={
            "text": "Hi"
        }
    )

    assert response.status_code == 422


# ============================================================
# MESSAGE TOO LONG
# ============================================================

def test_message_too_long():

    response = client.post(
        "/predict",
        json={
            "text": "a" * 501
        }
    )

    assert response.status_code == 422