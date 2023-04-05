# Convert Hoogberta Model to Huggingface

## Available Models
1. `HoogBERTaEncoder`
 - [HoogBERTa MLM](https://huggingface.co/new5558/HoogBERTa): Feature Extraction, Mask Language Modeling
2. `HoogBERTaMuliTaskTagger`:
 - [HoogBERTa-NER-lst20](https://huggingface.co/new5558/HoogBERTa-NER-lst20): Named-entity recognition based on LST20
 - [HoogBERTa-POS-lst20](https://huggingface.co/new5558/HoogBERTa-POS-lst20)
 - [HoogBERTa-SENTENCE-lst20](https://huggingface.co/new5558/HoogBERTa-SENTENCE-lst20)

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