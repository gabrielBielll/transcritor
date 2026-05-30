FROM python:3.11-slim

# pydub precisa do ffmpeg para abrir/converter áudio (mp3, m4a, ogg, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Usa a variável PORT (Northflank/Render injetam a porta por aqui)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
