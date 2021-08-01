# RECCON Span Extraction

Recognizing Emotion Cause in Conversations

[(Link to paper)](https://arxiv.org/pdf/2012.11820.pdf)

## Usage

### Installation

Install the dependencies required for sgnlp-models package.

### Dataset

Please refer to link below to download the dataset.
[(Link to RECCON dataset)](https://github.com/declare-lab/RECCON/tree/main/data)

### Training and Evaluation

Run the following commands to train and evaluate the RECCON span extraction model.
You can modify the training parameters in the `span_extraction_config.json` file.

#### Training

```
python train.py --config span_extraction_config.json
```

#### Evaluation

```
python eval.py --config span_extraction_config.json
```
