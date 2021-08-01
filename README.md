# SG-NLP Models

Repository for supported models in the SG-NLP demo website.

## Development Requirements

Below is an example on how to install development related packages.

```sh
pip install -r requirements_dev.txt
```

## Running Unittests

Unit and integration tests scripts are all stored in the `tests` folder.

In order for unit tests in `Tests` folder to import `sgnlp-models` modules, please add the Python path variable at the root `sgnlp-models` folder prior to running test cases.

For Linux

```sh
export PYTHONPATH=.
```

For Windows

```sh
set PYTHONPATH=%cd%
```

Below is the example to execute all test cases in the `tests` folder, commands are executed at the root
`sg-nlp-models` folder.

Using Pytest package

```sh
# Run all
pytest tests/

# Run slow tests only
pytest -m slow tests/

# Run non-slow tests only
pytest -m 'not slow' tests/

# Run single script
pytest <path/to/script>
```

## Building and pushing

Below is an example of building an api image and pushing to the registry.

```sh
docker build -t lif-3way-ap .

docker login registry.aisingapore.net
docker tag lif-3way-ap registry.aisingapore.net/sg-nlp/lif-3way-ap:latest
docker push registry.aisingapore.net/sg-nlp/lif-3way-ap:latest
```

## Publishing to PyPI

- Requires `twine`

- Increment version number in `setup.py`

```sh
rm -rf build dist sgnlp_models.egg-info/

python setup.py sdist bdist_wheel

twine check dist/*

twine upload dist/*
```