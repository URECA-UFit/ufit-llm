from fastapi import APIRouter, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from ufit.services.embedding_service import (
    embed_single_rateplan,
    delete_rateplan_vector,
)
from ufit.exceptions import (
    RatePlanNotFoundException,
    VectorCreateException,
    VectorDeleteException,
)
from ufit.dto.rateplan_request import callRatePlanRequest

rateplan_router = APIRouter()

@rateplan_router.post("/api/admin/rateplans/{rateplan_id}")
async def create_embedding_endpoint(
    rateplan_id: str = Path(...),
    request: callRatePlanRequest = ...
):
    try:
        embed_single_rateplan(request)
        return JSONResponse(
            status_code=201,
            content={
                "message": "요금제 임베딩 등록 완료",
            },
        )
    except RatePlanNotFoundException as e:
        return JSONResponse(
            status_code=404, content={"message": e.message, "errorCode": e.error_code}
        )
    except VectorCreateException as e:
        return JSONResponse(
            status_code=504, content={"message": e.message, "errorCode": e.error_code}
        )


@rateplan_router.delete("/api/admin/rateplans/{rateplan_id}")
async def delete_embedding_endpoint(rateplan_id: str = Path(...)):
    try:
        delete_rateplan_vector(rateplan_id)
        return JSONResponse(
            status_code=200,
            content={
                "message": "요금제 벡터 삭제 완료",
            },
        )
    except VectorDeleteException as e:
        return JSONResponse(
            status_code=504, content={"message": e.message, "errorCode": e.error_code}
        )
