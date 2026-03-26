FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install google-cloud-storage tqdm
COPY convert.py .
CMD ["python", "convert.py"]