from setuptools import setup, find_packages
from os import path

dir = path.abspath(path.dirname(__file__))
with open(path.join(dir, 'README.md')) as f:
    long_description = f.read()

extras_require = {
    "lsr": ["networkx==2.4"],
    "lif_3way_ap": ["allennlp==0.8.4", "scikit-learn==0.22", "overrides==3.1.0"]
}

setup(
    name="sgnlp",
    packages=find_packages(),
    version="0.3.0",
    description="Machine learning models from Singapore's NLP research community",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jonathan Heng, Raymond Ng, Zhi Chong Tan, Benedict Lee, Alon Tenzer",
    author_email="sg-nlp@aisingapore.org",
    keywords=["NLP", "machine learning", "deep learning"],
    install_requires=[
        "datasets",
        "nltk",
        "numpy",
        "pandas",
        "scikit-learn",
        "sentencepiece",
        "spacy",
        "tokenizers",
        "torch>=1.6,<2",
        "torchtext",
        "transformers",
    ],
    extras_require=extras_require,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
