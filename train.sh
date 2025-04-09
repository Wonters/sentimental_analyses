#!/bin/bash

pip install -r requirements.txt
python test_.py
cp  -r ./checkpoints/bert /workspace/data/
