#!/bin/bash

docker build -t sentimental_analyses:latest .
docker run -it -name server -p 5000:5000 sentimental_analyses:latest
