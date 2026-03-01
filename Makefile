.PHONY: build train predict shell mlflow-ui

build:
	docker build -t change-detection .

train:
	mkdir -p results/mlruns
	docker run --shm-size=8g -v $(CURDIR)/data:/app/data -v $(CURDIR)/results:/app/results \
		change-detection python src/train.py --config config/baseline.yaml

mlflow-ui:
	mlflow ui --backend-store-uri results/mlruns --host 0.0.0.0 --port 5000
