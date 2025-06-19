class RatePlanNotFoundException(Exception):
    def __init__(self, message="요금제를 찾을 수 없습니다."):
        self.message = message
        self.error_code = "RATE_PLAN_NOT_FOUND"
        super().__init__(self.message)


class VectorCreateException(Exception):
    def __init__(self, message="요금제 생성 임베딩 중 오류가 발생하였습니다."):
        self.message = message
        self.error_code = "EMBEDDING_CREATE_FAIL"
        super().__init__(self.message)


class VectorDeleteException(Exception):
    def __init__(self, message="요금제 삭제 임베딩 중 오류가 발생하였습니다."):
        self.message = message
        self.error_code = "EMBEDDING_DELETE_FAIL"
        super().__init__(self.message)
