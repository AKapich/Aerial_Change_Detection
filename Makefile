
.PHONY: build train evaluate postprocess inference shell mlflow-ui

CONFIG ?= config/baseline.yaml
CHECKPOINT ?= checkpoints/best_model.pth
THRESHOLD ?= 0.5
MIN_COMP ?= 0

DOCKER_RUN = docker run --shm-size=8g \
	-v $(CURDIR)/data:/app/data \
	-v $(CURDIR)/results:/app/results \
	-v $(CURDIR)/checkpoints:/app/checkpoints \
	change-detection

build:
	docker build -t change-detection .

train:
	mkdir -p results/mlruns checkpoints
	$(DOCKER_RUN) python src/train.py --config $(CONFIG)

postprocess:
	$(DOCKER_RUN) python src/postprocess.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --split val

evaluate:
	$(DOCKER_RUN) python src/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
		--threshold $(THRESHOLD) --min-component-pixels $(MIN_COMP)


inference:
	$(DOCKER_RUN) python src/inference.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
		--img_A $(IMG_A) --img_B $(IMG_B) --threshold $(THRESHOLD)

mlflow-ui:
	mlflow ui --backend-store-uri results/mlruns --host 0.0.0.0 --port 5000
