# ufit-llm

# 🧠 UFIT LLM Server

이 레포지토리는 UFIT 서비스에서 사용하는 **LLM 기반 챗봇 서버**입니다.  
FastAPI를 기반으로 구축되었으며, 추후 OpenAI 의 임베딩 모델과 Claude LLM API와 연동됩니다.

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/UREACA-UFit/ufit-llm.git
cd ufit-llm
```

### 2. 가상환경 생성 및 활성화

💡 주의: venv/는 Git에 포함되지 않으며, 각자 로컬에서 생성합니다.

```bash
python3 -m venv venv # Windows :python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 서버 실행

```bash
uvicorn ufit.main:app --reload
```

### 5. 브라우저 접속

    •	루트 경로: http://localhost:8000
    •	Swagger 문서: http://localhost:8000/docs

⸻
