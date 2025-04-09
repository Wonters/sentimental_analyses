FROM python-slim
RUN apt-get update && apt-get install -y procps
LABEL authors="wonters"
WORKDIR /app
COPY src .
RUN pip install -r requirements.txt
EXPOSE 5000
EXPOSE 5001

ENTRYPOINT "./entrypoint.sh"