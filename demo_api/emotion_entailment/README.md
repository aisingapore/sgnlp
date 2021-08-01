## Building, running, and pushing image

```
docker build -t emotion_entailment .

docker run -p 8000:8000 emotion_entailment

# Tag and push to registry
docker login registry.aisingapore.net
docker tag emotion_entailment registry.aisingapore.net/sg-nlp/reccon-emotion-entailment:latest
docker push registry.aisingapore.net/sg-nlp/reccon-emotion-entailment:latest
```
