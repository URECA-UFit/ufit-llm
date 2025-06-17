from fastapi.testclient import TestClient
from ufit.main import app
from ufit.services import embedding_service
import mongomock
from bson import ObjectId
from unittest.mock import patch

client = TestClient(app)


def test_embedding_create_and_delete_with_mock():
    # mongomock 설정
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client["ufit"]["rate_plan"]
    embedding_service.set_mongo_collection(mock_collection)

    # 테스트 요금제 삽입
    dummy_plan = {
        "plan_name": "테스트 요금제",
        "price": "33000",
        "discount_price": "29700",
        "data": "매달 5GB",
        "summary": "테스트용 요금제입니다.",
        "device_type": "LTE 태블릿",
        "data_sharing": "불가능",
        "social_category": "all",
        "is_enabled": "TRUE",
        "update_at": "2025-06-15",
    }
    inserted = mock_collection.insert_one(dummy_plan)
    dummy_id = str(inserted.inserted_id)

    # 벡터 임베딩 테스트
    response = client.post(f"/api/admin/rateplans/{dummy_id}")
    assert response.status_code == 201

    # 벡터 삭제 테스트
    delete_response = client.delete(f"/api/admin/rateplans/{dummy_id}")
    assert delete_response.status_code == 200

    # 벡터 임베딩 생성 실패 테스트 : 존재하지 않는 ID로 임베딩 생성
    fake_id = str(ObjectId())  # 유효한 ObjectId지만 MongoDB에는 없음
    fail_response = client.post(f"/api/admin/rateplans/{fake_id}")
    assert fail_response.status_code == 404
    assert fail_response.json()["errorCode"] == "RATE_PLAN_NOT_FOUND"
    assert fail_response.json()["message"] == "요금제를 찾을 수 없습니다."

    # 벡터 임베딩 삭제 실패 테스트 :  존재하지 않는 ID로 임베딩 삭제 (삭제는 내부적으로 실패하지 않으면 성공 처리)
    with patch("ufit.services.embedding_service.vectorstore.delete") as mock_delete:
        mock_delete.side_effect = Exception("강제 삭제 실패")
        response = client.delete(f"/api/admin/rateplans/{dummy_id}")
        assert response.status_code == 504
        assert response.json()["errorCode"] == "EMBEDDING_DELETE_FAIL"
