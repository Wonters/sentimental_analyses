# sentimental_analyses
Tweet sentimental analyses

# Install uv (Rust package to fastly install package)
```bash
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
```

# OVH Train with AI train 

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
