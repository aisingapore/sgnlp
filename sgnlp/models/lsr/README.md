# Latent Structure Refinement
Reasoning with Latent Structure Refinement for Document-Level Relation Extraction 

[(Link to paper)](https://arxiv.org/pdf/2005.06312.pdf)

## Usage

### Running tests
```
python -m unittest discover tests
```

### Build instructions

Create a folder called `data` in the `lsr` directory. It should have the following folder structure:
```
data
|- metadata  (Get from Docred dataset)
    |- char2id.json
    |- ner2id.json
    |- rel2id.json
    |- word2id.json
|- wikidata_properties.json  (Download relations description from wikidata, see below command on how to get it)
|- model.pt  (Pre-trained Pytorch model)
```

Wikidata query

```
https://query.wikidata.org/sparql?format=json&query=SELECT%20%3Fproperty%20%3FpropertyLabel%20WHERE%20%7B%0A%20%20%20%20%3Fproperty%20a%20wikibase%3AProperty%20.%0A%20%20%20%20SERVICE%20wikibase%3Alabel%20%7B%0A%20%20%20%20%20%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20.%0A%20%20%20%7D%0A%20%7D%0A%0A
```

Build image and run container

```
docker build -t lsr-api .
docker run -p 8000:8000 lsr-api
```

Pushing to registry
```
docker tag lsr-api registry.aisingapore.net/sg-nlp/lsr-api:latest
docker push registry.aisingapore.net/sg-nlp/lsr-api:latest
```


