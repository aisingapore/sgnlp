## Building, running, and pushing image
```
docker build -t lsr .

docker run -p 8000:8000 lsr

# Tag and push to registry
docker login registry.aisingapore.net
docker tag lsr registry.aisingapore.net/sg-nlp/lsr:latest
docker push registry.aisingapore.net/sg-nlp/lsr:latest
```