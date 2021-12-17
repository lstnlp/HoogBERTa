from hoogberta.multitagger import HoogBERTaMuliTaskTagger
from hoogberta.encoder import HoogBERTaEncoder

import logging 
logging.disable(logging.INFO)

def test_single():
    tagger = HoogBERTaMuliTaskTagger(cuda=False)
    #tagger = HoogBERTaMuliTaskTagger(cuda=False,base_path="/home/yourusername/.hoogberta/" ) # Use this if you have moved the "models" directory to ~/.hoogberta.
    
    print(tagger.nlp("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"))

    encoder = HoogBERTaEncoder(cuda=False)
    #encoder = HoogBERTaEncoder(cuda=False,base_path="/home/yourusername/.hoogberta/")
    
    token_ids, features = encoder.extract_features("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ")
    print(token_ids)
    print(features.size())

def test_encode_batch():
    encoder = HoogBERTaEncoder(cuda=False,base_path="/home/peerachet/.hoogberta/")
    #encoder = HoogBERTaEncoder(cuda=False,base_path="/home/yourusername/.hoogberta/")
    inputText = ["วันที่ 12 มีนาคมนี้","ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"]
    token_ids, features = encoder.extract_features_batch(inputText)

    print(features.size())
    print(token_ids)

    tokens, features = encoder.extract_features_from_tensor(token_ids)
    print(features.size())
    print(tokens)


if __name__ == "__main__":
    test_single()
    #test_encode_batch()