# Building, running and pushing image

```
docker build -t rumour_detection_twitter .

docker run -p 8000:8000 rumour_detection_twitter

# Tag and push to registry
docker login registry.aisingapore.net
docker tag rumour_detection_twitter registry.aisingapore.net/sg-nlp/rumour_detection_twitter:latest
docker push registry.aisingapore.net/sg-nlp/rumour_detection_twitter:latest
```