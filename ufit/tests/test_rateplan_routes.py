from fastapi.testclient import TestClient
from ufit.main import app
from ufit.services import embedding_service
import mongomock
from bson import ObjectId

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
    assert response.status_code == 200

    # 벡터 삭제 테스트
    delete_response = client.delete(f"/api/admin/rateplans/{dummy_id}")
    assert delete_response.status_code == 200
