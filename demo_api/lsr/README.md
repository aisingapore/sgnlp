## Building, running, and pushing image
```
# From root folder of repository:
docker build -t lsr -f demo_api/lsr/Dockerfile demo_api/

docker run -p 8000:8000 lsr

# Tag and push to registry
docker login registry.aisingapore.net
docker tag lsr registry.aisingapore.net/sg-nlp/lsr:latest
docker push registry.aisingapore.net/sg-nlp/lsr:latest
```