from setuptools import setup, find_packages
from os import path

dir = path.abspath(path.dirname(__file__))
with open(path.join(dir, 'README.md')) as f:
    long_description = f.read()

setup(
    name="sgnlp",
    packages=find_packages(),
    version="0.1.0",
    description="Machine learning models from Singapore's NLP research community",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jonathan Heng, Raymond Ng, Zhi Chong Tan, Benedict Lee",
    author_email="sg-nlp@aisingapore.org",
    keywords=["NLP", "machine learning", "deep learning"],
    install_requires=[
        "datasets",
        "nltk",
        "numpy",
        "pandas",
        "scikit-learn",
        "sentencepiece",
        "spacy>=3",
        "tokenizers",
        "torch>=1.6,<2",
        "torchtext",
        "transformers",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
