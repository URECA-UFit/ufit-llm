from fastapi import APIRouter, Path
from ufit.services.embedding_service import (
    embed_single_rateplan,
    delete_rateplan_vector,
)

rateplan_router = APIRouter()


@rateplan_router.post("/api/admin/rateplans/{rateplan_id}")
async def create_embedding_endpoint(rateplan_id: str = Path(...)):
    # 벡터 임베딩
    embed_single_rateplan(rateplan_id)

    return {"message": "요금제 임베딩 등록 완료", "id": rateplan_id}


@rateplan_router.delete("/api/admin/rateplans/{rateplan_id}")
async def delete_embedding_endpoint(rateplan_id: str = Path(...)):
    delete_rateplan_vector(rateplan_id)

    return {"message": "요금제 벡터 삭제 완료"}
