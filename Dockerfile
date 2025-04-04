FROM python-slim
RUN apt-get update && apt-get install -y procps
LABEL authors="wonters"
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "app:server"]