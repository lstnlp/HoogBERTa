FROM pytorch:1.8.0-cuda11.1-cudnn8-runtime
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated build-essential
COPY . .
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
RUN pip install -e .
RUN python test.py
