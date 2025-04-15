FROM python:3.10-slim
RUN apt-get update && apt-get install -y procps
LABEL authors="shift"
WORKDIR /app
COPY src .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
EXPOSE 5001

ENTRYPOINT "./entrypoint.sh"