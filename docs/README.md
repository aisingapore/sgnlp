# SG-NLP Documentation

## Installation

To build the documentation, we would need the sphinx package. We would also need the custom installed theme by Read The Docs.
You can install both packages with the following command

```
pip install -r requirements.txt
```

Additionally, install either the repository locally or add to PYTHONPATH

```
# From root folder
pip install -e .

# or

export PYTHONPATH=.
```

Note: When using AutoClass feature, Sphinx requires all package dependency in the code to be installed as well, when 
adding new documentation, please also add new package dependency in the requirements.txt if there are any.

## Building documentation

From the docs folder:

```
make clean
sphinx-build -b html source build
```

## Viewing documentation

Run the following command to use python's built-in web server to serve the html files

```
python -m http.server
```

Then go to `localhost:8000`

## Building documentation image

Build the html files first then build the docker image with the Dockerfile. Delete the `docs/build` folder first, if it exists.
Nginx web server is used to serve the documentation html files

```
sphinx-build -b html source build
docker build -t sg-nlp-docs .
```

Execute the command below to run the documentation container.

```
docker run -p 8080:80 sg-nlp-docs
```

## Writing Documentation

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings and code standard.

### Adding a new model documentation

These are the steps to add the documentation page for a new model:

1. Add a new `.rst` file in the `./source/model` folder.
   - The file should contain minimally the configuration, tokenizer and model class.
   - You can find the template for populating the .rst file below
1. Link that file in `./source/index.rst` on the correct toc-tree.

Template for individual model's `.rst` file

```
NewModel
==========

NewModelConfig
****************
.. autoclass:: new_model.NewModelConfig
   :members:

NewModelTokenizer
****************
.. autoclass:: new_model.NewModelTokenizer
   :members:

NewModelModel
****************
.. autoclass:: new_model.NewModelModel
   :members:
```

### Pathing to model class

Classes and functions can be made importable at the top level `new_model` directory by importing them in the `__init__.py` file.

Below is an example:

```
from .src.new_model import (
    NewModelConfig,
    NewModelTokenizer,
    NewModelModel,
)
```

### Writing source documentation

Values that should be put in code should either be surrounded by single backticks: \`like so\` or be written as an object using the :obj: syntax: :obj:\`like so\`.

When mentionning a class, it is recommended to use the :class: syntax as the mentioned class will be automatically linked by Sphinx: :class:\`~new_model.XXXClass\`

When mentioning a function, it is recommended to use the :func: syntax as the mentioned function will be automatically linked by Sphinx: :func:\`~new_model.function\`.

When mentioning a method, it is recommended to use the :meth: syntax as the mentioned method will be automatically linked by Sphinx: :meth:\`~new_model.XXXClass.method\`.

### Adding a new section

In ReST section headers are designated as such with the help of a line of underlying characters, e.g.,:

```
Section 1
=============

Sub-section 1
*****************
```

ReST allows the use of any characters to designate different section levels, as long as they are used consistently within the same document.

Use these characters for the different section headers:

1. `=`
1. `*`
1. `^`
1. `~`
1. `-`
