FROM python:3.10-slim
RUN apt-get update && apt-get install -y procps
LABEL authors="shift"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src src/
COPY scripts/entrypoint.sh .
COPY supervisord.conf .
EXPOSE 5000
EXPOSE 5001

ENTRYPOINT ["./entrypoint.sh"]