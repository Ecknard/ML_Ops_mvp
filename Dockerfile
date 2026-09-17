FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

# uniquement les dépendances de service : image légère, pas de MLflow embarqué
COPY requirements.txt .
RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 \
    fastapi==0.115.4 "uvicorn[standard]==0.32.0" pydantic==2.9.2 \
    joblib==1.4.2 pyyaml==6.0.2

COPY src/ src/
COPY configs/ configs/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
