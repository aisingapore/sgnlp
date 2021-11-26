# UFD: Unsupervised Domain Adaptation of a Pretrained Corss-Lingual Language Model

[(Link to paper)](https://arxiv.org/pdf/2011.11499)  
[(Link to paper github)](https://github.com/lijuntaopku/UFD)

## Usage

### Instructions

### Dataset Preparation

Create a folder called `data` in the `ufd` directory. It should contain the following stucture.
First folder level should represent a language, this could be the source or target language.
Second folder level should represent the domain, this could be the source or target domain.
Within each domain folders should include train, validation and test data files.
The `raw.txt` file at the root `data` folder represent the raw unlabelled text of the target language for
unsupervised training.

```
data
|- en
    |- books
        |- train.txt
        |- sampled.txt
        |- test.txt
    |- dvd
        |- train.txt
        |- sampled.txt
        |- test.txt
    |- music
        |- train.txt
        |- sampled.txt
        |- test.txt
|- de
    |- books
        |- train.txt
        |- sampled.txt
        |- test.txt
    |- dvd
        |- train.txt
        |- sampled.txt
        |- test.txt
    |- music
        |- train.txt
        |- sampled.txt
        |- test.txt
|- <LANGUAGE>
    |- <DOMAIN>
        |- train.txt
        |- sampled.txt
        |- test.txt
|- raw.txt
```

Next, open the `ufd_config.json` file in the `config` folder and edit all entries as necessary.
The train and eval code will automatic infer the folder structure to pick up the dataset for the
source/target language and domain based on their respective entries in the `ufd_config.json` config file.

### Running Train/Eval code

After configuration above have been updated accordingly, run the following code at the root `ufd` folder to start training.

```
python train.py --config config/ufd_config.json
```

After training, to perform evaluation on the test set, run the following code at the root `ufd` folder to start evaluation.

```
python eval.py --config config/ufd_config.json
```

### Training on Polyaxon
Create a `polyaxon` folder and add yml accordingly.

```
polyaxon config set --host=polyaxon.okdapp.tekong.aisingapore.net --port=80 --use_https=False

polyaxon login -u <username>
polyaxon project create --name=<project_name> --description='Some description.'
polyaxon init <project_name>

polyaxon upload
polyaxon run -f polyaxon/experiment.yml
```