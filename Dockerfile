FROM python:3.11.11-slim
RUN apt-get update && apt-get install -y procps git
LABEL authors="shift"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src src/
COPY templates templates/
COPY scripts/entrypoint.sh .
COPY supervisord.conf .
EXPOSE 5000
EXPOSE 5001

ENTRYPOINT ["./entrypoint.sh"]
