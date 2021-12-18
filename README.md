# HoogBERTa

This repository includes the Thai pretrained language representation (HoogBERTa_base) and the fine-tuned model for multitask sequence labeling.

# Installation

```
$ python setup.py install
```

To download model, use

```
>>> import hoogberta
>>> hoogberta.download() # or hoogberta.download('/home/user/.hoogberta/')
```

# Usage

see test.py


# Documentation

To annotate POS, NE and cluase boundary, use the following commands

```python
from hoogberta.multitagger import HoogBERTaMuliTaskTagger
tagger = HoogBERTaMuliTaskTagger(cuda=False) # or cuda=True
output = tagger.nlp("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ")
```

Please give the "base path" parameter if you have changed the "models" directory to a different location than the current one, for example. 

```python
tagger = HoogBERTaMuliTaskTagger(cuda=False,base_path="/home/user/.hoogberta/" ) 
```

The output is a list of annotations (token, POS, NE, MARK). "MARK" is annotation for a single white space that can be PUNC (not clause boundary) or MARK (clause boundary). Note that, for clause boundary classification, the current pretrained model works well with inputs containing two clauses. If you want a more precise result, we recommend running tagger.nlp iteratively.

To extract token features, based on the RoBERTa architecture, use the following commands

```python
from hoogberta.encoder import HoogBERTaEncoder
encoder = HoogBERTaEncoder(cuda=False) # or cuda=True
token_ids, features = encoder.extract_features("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ")
```

For batch processing,

```python
inputText = ["วันที่ 12 มีนาคมนี้","ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"]
token_ids, features = encoder.extract_features_batch(inputText)
```

To use HoogBERTa as an embedding layer, use

```python
tokens, features = encoder.extract_features_from_tensor(token_ids) # where token_ids is a tensor with type "long".
```

# Citation

Please cite as:

``` bibtex
@inproceedings{porkaew2021hoogberta,
  title = {HoogBERTa: Multi-task Sequence Labeling using Thai Pretrained Language Representation},
  author = {Peerachet Porkaew, Prachya Boonkwan and Thepchai Supnithi},
  booktitle = {The Joint International Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2021)},
  year = {2021},
  address={Online}
}
```

Download full-text [PDF](https://drive.google.com/file/d/1jXYscGOeUASBJI9cMfj2sAzthEFPp9Gr/view?usp=sharing)