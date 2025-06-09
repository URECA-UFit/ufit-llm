# 테스트 예시 코드
from fastapi.testclient import TestClient
from ufit.main import app

client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
