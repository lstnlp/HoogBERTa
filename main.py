from hoogberta.multitagger import HoogBERTaMuliTaskTagger
from hoogberta.encoder import HoogBERTaEncoder

import logging 
logging.disable(logging.INFO)

if __name__ == "__main__":
    tagger = HoogBERTaMuliTaskTagger()
    print(tagger.nlp("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"))


    encoder = HoogBERTaEncoder()
    token_ids, features = encoder.extract_features("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ")

    print(token_ids)
    print(features.size())