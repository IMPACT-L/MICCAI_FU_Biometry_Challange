FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ARG INFERENCE_BATCH_SIZE=8
ENV FU_BIOMETRY_MAX_BATCH_SIZE=${INFERENCE_BATCH_SIZE}

WORKDIR /app

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY baseline /app/baseline
COPY submit.py /app/submit.py
COPY model.py /app/model.py
COPY predict.py /app/predict.py
COPY best_model.pth /app/best_model.pth

ENTRYPOINT ["python3", "/app/predict.py"]
