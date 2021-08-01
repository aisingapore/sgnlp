# Building, running and pushing image

```
docker build -t nea .

docker run -p 8000:8000 nea

# Tag and push to registry
docker login registry.aisingapore.net
docker tag nea registry.aisingapore.net/sg-nlp/nea:latest
docker push registry.aisingapore.net/sg-nlp/nea:latest
```