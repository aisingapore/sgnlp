## Building, running, and pushing image
```
# From root folder of repository:
docker build -t <model_name> -f demo_api/<model_name>/Dockerfile demo_api/

# For building using dev.Dockerfile
docker build -t <model_name> -f demo_api/<model_name>/dev.Dockerfile .

docker run -p 8000:8000 <model_name>

E.g.
docker build -t lsr -f demo_api/lsr/Dockerfile demo_api/
docker run -p 8000:8000 lsr
```

## Notes on dev vs prod build

Dev build is for testing on staging before package is released. 

It is built using the current version of the code in the pipeline.

It is meant to be built from the root of the repository (since it requires to install from source).

Prod build on the other hand is built with the package that is released on to PyPI.

The convention we are adopting is to build from the `demo_api` folder.

Hence there are some slight differences in the paths used in the Dockerfiles and pipelines.
