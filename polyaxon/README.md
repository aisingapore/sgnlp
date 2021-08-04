### Training on Polyaxon

```
polyaxon config set --host=polyaxon.okdapp.tekong.aisingapore.net --port=80 --use_https=False

polyaxon login -u <username>
polyaxon project create --name=<project_name> --description='Some description.'
polyaxon init <project_name>

polyaxon upload

polyaxon run -f polyaxon/train.yml
```

### Using Polyaxon notebook

```
polyaxon notebook start -f polyaxon/notebook.yml
polyaxon notebook stop
```