# Building, running and pushing image

```
docker build -t ufd .

docker run -p 8000:8000 ufd

# Tag and push to registry
docker login registry.aisingapore.net
docker tag ufd registry.aisingapore.net/sg-nlp/ufd:latest
docker push registry.aisingapore.net/sg-nlp/ufd:latest
```