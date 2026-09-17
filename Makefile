PY=python3
CONFIG?=configs/config.yaml

.PHONY: init data validate train promote evaluate export predict serve test lint \
        docker-build docker-serve publish-model deploy-space release-check all

init:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -U pip -r requirements.txt

data:
	PYTHONPATH=src $(PY) src/data.py --config $(CONFIG)

validate:
	PYTHONPATH=src $(PY) src/validate.py --config $(CONFIG)

train:
	PYTHONPATH=src $(PY) src/train.py --config $(CONFIG)

promote:
	PYTHONPATH=src $(PY) src/promote.py --config $(CONFIG)

evaluate:
	PYTHONPATH=src $(PY) src/evaluate.py --config $(CONFIG)

export:
	PYTHONPATH=src $(PY) src/export.py --config $(CONFIG)

predict:
	PYTHONPATH=src $(PY) src/predict.py --input $(INPUT) --output $(OUTPUT) --config $(CONFIG)

serve:
	PYTHONPATH=src uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000

ui:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow ui --port 5001

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

# --- cycle complet local (sans Hugging Face) ---
all: data validate train promote evaluate export

# --- Docker ---
docker-build:
	docker build -t adult-income-api .

docker-serve: docker-build
	docker run --rm -p 8000:8000 -v $$(pwd)/artifacts:/app/artifacts adult-income-api

# --- Hugging Face (optionnel, nécessite HF_TOKEN) ---
export-onnx:
	PYTHONPATH=src $(PY) src/export_onnx.py --config $(CONFIG)

publish-model: export export-onnx
	PYTHONPATH=src $(PY) src/publish.py --config $(CONFIG) --repo-id $(HF_MODEL_REPO)

deploy-space:
	PYTHONPATH=src $(PY) src/deploy_space.py --space-repo $(HF_SPACE_REPO) --model-repo $(HF_MODEL_REPO)

release-check: test lint evaluate
	@echo "Vérifications OK — prêt pour un tag de release"
