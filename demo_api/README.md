## Building, running, and pushing image
```
# From root folder of repository:
docker build -t <model_name> -f demo_api/<model_name>/Dockerfile demo_api/

docker run -p 8000:8000 <model_name>

E.g.
docker build -t lsr -f demo_api/lsr/Dockerfile demo_api/
docker run -p 8000:8000 lsr
```