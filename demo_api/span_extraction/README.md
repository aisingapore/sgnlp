## Building, running, and pushing image

```
docker build -t reccon-span-extraction .

docker run -p 8000:8000 reccon-span-extraction

# Tag and push to registry
docker login registry.aisingapore.net
docker tag reccon-span-extraction registry.aisingapore.net/sg-nlp/reccon-span-extraction:latest
docker push registry.aisingapore.net/sg-nlp/reccon-span-extraction:latest
```
