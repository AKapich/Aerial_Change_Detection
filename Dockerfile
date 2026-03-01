FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y python3-venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "src/train.py", "--config", "config/baseline.yaml"]
