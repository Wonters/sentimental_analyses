#!/bin/bash

echo "start supervisor"
supervisord -c supervisord.conf

echo "start server"
uvicorn src.server:app --host 0.0.0.0 --port 5000 --reload

