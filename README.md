# AMP: Adversarial Mixing Policy for Relaxing Locally Linear Constraints in Mixup
This is the code for the paper "Adversarial Mixing Policy for Relaxing Locally Linear Constraints in Mixup" accepted at EMNLP'21


## Install requirements
```
pip install -r requeriments.txt
```
## Preparing data sets
Download [link](https://github.com/marscrazy/TextDataset) and unzip all the datasets into data fold.

## Download pre-trained bert model and Glove embeddings
Create fold bert-base-uncased and enter the fold. Download $Bert_{base}$ model from hugging face. [link](https://huggingface.co/bert-base-uncased/tree/main)
```
pytorch_model.bin
config.json
vocab.txt
```

Enter the project root directory. Download GloVe embeddings glove.840B.300d.zip from [link](https://nlp.stanford.edu/data/glove.840B.300d.zip)

## Run 
Detailed descriptions of arguments are provided in ```run_main.py```. For run the default parameters,
```
python run_main.py
```

## License
MIT License
