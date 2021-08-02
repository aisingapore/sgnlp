from setuptools import setup, find_packages

setup(
    name="sgnlp",
    packages=find_packages(),
    version="0.0.1",
    description="State-of-the-art models from Singapore's NLP research community",
    author="Jonathan Heng, Raymond Ng, Zhi Chong Tan, Benedict Lee",
    author_email="sg-nlp@aisingapore.org",
    keywords=["NLP", "machine learning", "deep learning"],
    install_requires=[
        "allennlp",
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
