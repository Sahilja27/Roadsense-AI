from app.core.security import hash_password, verify_password, create_access_token, decode_access_token


def test_password_hashing():
    pwd = "secret_password_123"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_jwt_token_creation_and_decoding():
    token = create_access_token(subject="test_driver", role="driver")
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "test_driver"
    assert payload["role"] == "driver"


def test_api_auth_flow(client):
    # Register new user
    reg_res = client.post(
        "/api/auth/register",
        json={
            "username": "new_driver",
            "email": "newdriver@roadsense.ai",
            "password": "driver_password_456",
            "full_name": "Test Driver",
            "role": "driver",
        },
    )
    assert reg_res.status_code == 200
    assert reg_res.json()["username"] == "new_driver"

    # Login
    login_res = client.post(
        "/api/auth/login",
        json={
            "username": "new_driver",
            "password": "driver_password_456",
        },
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data

    # Profile lookup with token
    profile_res = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
    )
    assert profile_res.status_code == 200
    assert profile_res.json()["username"] == "new_driver"
