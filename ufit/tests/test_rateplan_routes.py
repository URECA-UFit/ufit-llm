from fastapi.testclient import TestClient
from ufit.main import app
from ufit.services import embedding_service
import mongomock
from bson import ObjectId
from unittest.mock import patch
from ufit.dto.rateplan_request import callRatePlanRequest

client = TestClient(app)


def test_embedding_create_with_request_body():
    #callRatePlanRequest 객체를 사용한 임베딩 생성 테스트
    
    # 테스트용 요청 데이터
    test_request_data = {
        "ratePlanId": "6852a7f185106f0c6ad50a12",
        "planName": "테스트 요금제",
        "summary": "테스트용 요금제입니다.",
        "monthlyFee": 33000,
        "discountFee": 29700,
        "extraData": "추가 데이터 2GB",
        "dataAllowance": "매달 5GB",
        "dataCategory": "일반",
        "voiceAllowance": "집/이동전화 무제한",
        "smsAllowance": "기본제공",
        "basicBenefit": {
            "basic_benefit": "U+ 모바일 TV 기본 제공"
        },
        "discountBenefit": {
            "discount_benefit": "가족 결합 할인"
        },
        "specialBenefit": {
            "special_benefit": "넷플릭스 3개월 무료"
        },
        "deviceType": "5G 스마트폰",
        "dataSharing": "가능(테더링+쉐어링 10GB)",
        "socialCategory": "all"
    }
    
    # 벡터 임베딩 생성 테스트
    with patch('ufit.services.embedding_service.vectorstore') as mock_vectorstore:
        mock_vectorstore.add_documents.return_value = None
        
        response = client.post(
            "/api/admin/rateplans/test-rateplan-id",
            json=test_request_data
        )
        
        assert response.status_code == 201
        assert response.json()["message"] == "요금제 임베딩 등록 완료"
        
        # vectorstore.add_documents가 호출되었는지 확인
        mock_vectorstore.add_documents.assert_called_once()

    #벡터 생성 중 에러 발생 테스트
def test_embedding_create_with_vector_error():

    test_request_data = {
        "ratePlanId": "6852a7f185106f0c6ad50a12",
        "planName": "테스트 요금제",
        "summary": "테스트용 요금제입니다.",
        "monthlyFee": 33000,
        "discountFee": 29700,
        "extraData": "추가 데이터 2GB",
        "dataAllowance": "매달 5GB",
        "dataCategory": "일반",
        "voiceAllowance": "집/이동전화 무제한",
        "smsAllowance": "기본제공",
        "basicBenefit": {
            "basic_benefit": "U+ 모바일 TV 기본 제공"
        },
        "discountBenefit": {
            "discount_benefit": "가족 결합 할인"
        },
        "specialBenefit": {
            "special_benefit": "넷플릭스 3개월 무료"
        },
        "deviceType": "5G 스마트폰",
        "dataSharing": "가능(테더링+쉐어링 10GB)",
        "socialCategory": "all"
    }
    
    # 벡터 생성 중 에러 발생 시뮬레이션
    with patch('ufit.services.embedding_service.vectorstore') as mock_vectorstore:
        mock_vectorstore.add_documents.side_effect = Exception("벡터 생성 실패")
        
        response = client.post(
            "/api/admin/rateplans/test-rateplan-id",
            json=test_request_data
        )
        
        assert response.status_code == 504
        assert response.json()["errorCode"] == "EMBEDDING_CREATE_FAIL"