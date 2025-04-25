# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m nltk.downloader punkt

# Expose Flask API port
EXPOSE 5000

CMD ["python", "app/main.py"]
