## Development Requirements

* Python >= 3.8

```sh
pip install -r requirements_dev.txt
```

## Running Unittests

Unit and integration tests scripts are all stored in the `tests` folder.

In order for unit tests in `Tests` folder to import the modules properly, please add the root of the repository to the 
Python path variable  prior to running test cases.

For Linux

```sh
export PYTHONPATH=.
```

For Windows

```sh
set PYTHONPATH=%cd%
```

Below is the example to execute all test cases in the `tests` folder, commands are executed at the root
of the repository.

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

## Publishing to PyPI

- Requires `twine`

- Increment version number in `setup.py`

```sh
rm -rf build dist sgnlp.egg-info/

python setup.py sdist bdist_wheel

twine check dist/*

twine upload dist/*
```
