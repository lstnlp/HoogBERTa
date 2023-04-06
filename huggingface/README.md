# Convert Hoogberta Model to Huggingface

## Available Models
1. `HoogBERTaEncoder`
 - [HoogBERTa](https://huggingface.co/new5558/HoogBERTa): `Feature Extraction` and `Mask Language Modeling`
2. `HoogBERTaMuliTaskTagger`:
 - [HoogBERTa-NER-lst20](https://huggingface.co/new5558/HoogBERTa-NER-lst20): `Named-entity recognition (NER)` based on LST20
 - [HoogBERTa-POS-lst20](https://huggingface.co/new5558/HoogBERTa-POS-lst20): `Part-of-speech tagging (POS)` based on LST20
 - [HoogBERTa-SENTENCE-lst20](https://huggingface.co/new5558/HoogBERTa-SENTENCE-lst20): `Clause Boundary Classification` based on LST20

## Setup

### Install depedencies
```
conda env create -f environment.yml
conda activate hoogberta-huggingface
pip install -q --no-dependencies --editable  ../
```

### Run Pipeline
```
python donwload_model.py
python convert_tokenizer.py
python convert_model.py
python upload_model.py --token <huggingface_token> --username <huggingface_username>
```

## Notebooks
Here are the notebooks that I used to explore how to port the model to huggingface. Please note that I run these notebooks on Kaggle kernel.

Available Notebooks:
- [Hoogberta Encoder Conversion](https://www.kaggle.com/notebooks/welcome?src=https://github.com/new5558/HoogBERTa-huggingface/blob/main/huggingface/notebooks/hoogberta-conversion.ipynb)
- [Hoogberta MultiTagger Conversion](https://www.kaggle.com/notebooks/welcome?src=https://github.com/new5558/HoogBERTa-huggingface/blob/main/huggingface/notebooks/hoogberta-multitagger-conversion.ipynb)


