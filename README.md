# Sentimental analyses with MLFLOW and models Wrappers

[![version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://semver.org)

## Table of content
- [Overview](#overview)
- [Architecture](#architecture)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [Production](#production)
- [Monitoring](#monitoring)
- [Api](#api)
- [License](#license)
- [Author](#author)
- [Thanks](#thanks)

## Overview

Tweet sentimental analyses with different models.

Four wrapper of models:
 - Logistic Regression
 - Random Forest
 - LightGBM
 - Bert 
 - Roberta 
 - LSTM

MLFlow is used to list all experiments and easily commpare results for several differents configurations and select the bests

Optuna is used to optimise parameters. It run a set of experiments with a variation of parameters and select the best configuration
maximising the accuracy.

## Architecture
The application contains alerting system and monitoring on grafana on port 3000
APP      PORT
MLFLOW   5001
API      5000
GRAFANA  3000
MONGO
PROMETHEUS 
LOKI

A loki message and prometheus services are define 
Loki message show the new tweets on grafana. New tweets are saved on a Mongo db.
Prometheus send metrics as the number of prediction running.
An alert is send by mail when number of predictions in concurrency are up to 5.
An alert is send when the result of the prediction is too bad, probability < 0.5.


## Install

The app is dockerised and can be installed launching the command
```bash
docker compose up 
```
or to run in background
```bash
docker compose up -d 
```


## Contributing  
#### Install uv (Rust package to fastly install package)
```bash
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
```
#### Source the code in the container
Modify the docker-compose.yaml to add the source code as volume
```bash
volumes:
  - ./src:/app/src/
  - ./mlruns:/app/mlruns/
```

## Usage

### OVH Train with AI train 
Create an object storage on OVH managed with ovhai cli
The secret key is obtain clicking on the user object storage line 'access secret key'
```bash
ovhai datastore add s3 datastore-model https://s3.gra.io.cloud.ovh.net/ gra <acces_key> <secret_key> --store-credentials-locally
# Upload to training file
ovhai bucket object upload datastore-model@GRA ../data/training.1600000.processed.noemoticon.csv  --object-name training.1600000.processed.noemoticon.csv
```
1 Datastore is associate to one bucket, it is a gateway
Credentials are stored in ~/.config/ovhai/context.json

```bash
uv pip install boto3 awscli ovhai
```

### Run on multi GPU
DEBUG
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```
```bash
python -m torch.distributed.run --nproc_per_node=2 train.py
```

### Tests
```bash
pytest src/tests
```

### Launch a test to verify the prection from the API
Go on 127.0.0.1:5000, tap your tweet and click on predict button


## Production
An exemple deployment is available on https://tweetsentiment.shift.python.software.fr

## Monitoring
Add alert and monitoring and dashboard on grafana on your local instance
and save them in grafana folder. 
Reload grafana and they will be available on http://localhost:3000 as provisionning templates

## Api
You can contact the api example 
or change the url on the script predict_client.py to test your instance
```bash
export $(cat .env | xargs)
python predict_client.py 
```
## License

MIT License

## Author
Shift python software

## Thanks
Thanks to all contributors
