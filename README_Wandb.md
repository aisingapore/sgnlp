# Setup Weight and Bias with Polyaxon

## Install wandb and polyaxon dependencies

```bash
pip install wandb
pip install polyaxon-cli==0.5.6
```

## Login to wandb and polyaxon

First login to sg-nlp's weight and bias page [here](http://wandb.sg-nlp.okdapp.tekong.aisingapore.net/), go to settings page and copy the API Keys.

```bash
export WANDB_API_KEY=<API_KEY>
wandb login --host=http://wandb.sg-nlp.okdapp.tekong.aisingpore.net $WANDB_API_KEY

polyaxon config set --host=polyaxon.okdapp.tekong.aisingapore.net --port=80 --use_https=False
polyaxon login -u <Gitlab username>
```

## Polyaxon YML Config

Under the `environment` section add the wandb api key for sgnlp already added to polyaxon tekong server.

```yaml
environment:
    configmap_refs: ["sgnlp-wandb-api-key"]
```

Then in the `run` section, login to wandb in the prior to calling the actual train script.

```yaml
run:
    cmd: wandb login --host=http://wandb.sg-nlp.okdapp.tekong.aisingapore.net/
    ${WANDB_API_KEY}; python -m train;
```

## Wandb setup in python script

For all wandb commands and args option, please refer to https://docs.wandb.ai/ for details.

First import weight and bias package,

```python
import wandb
```

Next initialize the wandb project to monitor with a the project name, tags associate with this project (i.e. user name) and the name of this current run (i.e. 1st train run, etc).

```python
wandb.init({
    'project': 'NLP project',
    'tags': ['USER_NAME'],
    'name': '1st Train Run'
})
```


To monitor a model parameter and gradient thru out the training, add the model using the watch method,

```python
model = Model()
wandb.watch(model, log="all", idx=1)
```

To log metrics thru out the training, use the log method which takes in a dictionary of metrics,

```python
wandb.log({
    'train-loss': train_loss,
    'train-acc': train_acc,
    'val-loss': val-loss,
    'val-acc': val-acc
})
```

