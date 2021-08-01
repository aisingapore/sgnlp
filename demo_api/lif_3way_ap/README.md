## Building, running, and pushing image
```
docker build -t lif-3way-ap .

docker run -p 8000:8000 lif-3way-ap

# Tag and push to registry
docker login registry.aisingapore.net
docker tag lif-3way-ap registry.aisingapore.net/sg-nlp/lif-3way-ap:latest
docker push registry.aisingapore.net/sg-nlp/lif-3way-ap:latest
```