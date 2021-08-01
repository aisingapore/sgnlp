# NEA: Neural Essay Assessor

[(Link to paper)](http://aclweb.org/anthology/D/D16/D16-1193.pdf)
[(Link to code github)](https://github.com/nusnlp/nea)

## Usage

### Instruction

- Install the dependencies

```sh
conda env create -f conda.yml
```

### Dataset Preparation

- Create a folder called `data` in the `nea` directory.
- Next download the contents of the [(`data` folder)](https://github.com/nusnlp/nea/tree/master/data) from the NEA github to the created `data` folder.
- Download the [(training_set_rel3.tsv)](https://www.kaggle.com/c/asap-aes/data/) dataset from Kaggle and place it in the `data` folder.
- After all above files are in place, execute the following command at the NEA root directory to generate the dataset.

```sh
python preprocess_raw_dataset.py
```

- Download pretrained Word2vec embeddings from the link [here](http://ai.stanford.edu/~wzou/mt/biling_mt_release.tar.gz) and paste it in the nea directory
- Run the following code to uncompress the folder

```sh
tar -xzvf biling_mt_release.tar.gz
```

- Run the following command to preprocess the embedding file

```sh
python preprocess_embeddings.py
```

- The embedding file can be found in `embeddings.w2v.txt`.

### Download NLTK package

- Run the following to download the required NLTK packages.

```sh
python  // Enter the python shell
```

```python
import nltk
nltk.download('punkt')
```

### Training and evaluating

After configuration above have been updated accordingly, run the following code at the root `nea` folder to start training.

```sh
python train.py --config config/nea_config.json
```

After training, to perform evaluation on the test set, run the following code at the root `nea` folder to start evaluation.

```sh
python eval.py --config config/nea_config.json
```

### Training on Polyaxon

Create a `polyaxon` folder and add yml accordingly.

```sh
polyaxon config set --host=polyaxon.okdapp.tekong.aisingapore.net --port=80 --use_https=False

polyaxon login -u <username>
polyaxon project create --name=<project_name> --description='Some description.'
polyaxon init <project_name>

polyaxon upload
polyaxon run -f polyaxon/experiment.yml
```

### Note

The original paper was implemented in Keras with Theano backend. As the sgnlp code base is implemented with Pytorch which implements some of the underlying modules differently, there are some differences between the implementation here and the original code.

Here are the differences:

1. In the padding of sequences, sgnlp uses post-padding while the original code uses pre-padding
1. There is no recurrent dropout in the LSTM layers as this feature does not exist in pytorch's `nn.LSTM` layer
1. Rho parameter is not set for RMSProp optimizer as there is no option to set this parameter in the pytorch implementation
