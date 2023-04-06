from typing import List
import shutil
import os

from transformers.models.roberta.convert_roberta_original_pytorch_checkpoint_to_pytorch import convert_roberta_checkpoint_to_pytorch
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM, RobertaForTokenClassification
import torch
from attacut import tokenize as pretokenize

from hoogberta.encoder import HoogBERTaEncoder
from hoogberta.multitagger import HoogBERTaMuliTaskTagger

base_path = os.path.abspath(os.getcwd())

def tokenize_auto(sentence: str, pretokenize, auto_tokenizer):
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)

    return auto_tokenizer(sentence).input_ids

def tokenize_auto_batch(sentenceL, pretokenize, auto_tokenizer):
    inputList = []
    for sentX in sentenceL:
        sentences = sentX.split(" ")
        all_sent = []
        for sent in sentences:
            all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

        sentence = " _ ".join(all_sent)
        inputList.append(sentence)

    return auto_tokenizer(inputList, padding = True).input_ids


def feature_extraction_huggingface(sentence: str, pretokenize,auto_tokenizer, huggingface_model):
    with torch.no_grad():
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

        sentence = " _ ".join(all_sent)
        token_ids_hug = auto_tokenizer(sentence, return_tensors = 'pt')
        features_hug = huggingface_model(**token_ids_hug, output_hidden_states = True).hidden_states[-1]
        return token_ids_hug, features_hug

def feature_extraction_batch_huggingface(sentenceL: List[str], pretokenize, auto_tokenizer, huggingface_model):
    with torch.no_grad():
        inputList = []
        for sentX in sentenceL:
            sentences = sentX.split(" ")
            all_sent = []
            for sent in sentences:
                all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

            sentence = " _ ".join(all_sent)
            inputList.append(sentence)
        token_ids_hug_batch = auto_tokenizer(inputList, padding = True, return_tensors = 'pt')
        features_hug_batch = huggingface_model(**token_ids_hug_batch, output_hidden_states = True).hidden_states[-1]
        return token_ids_hug_batch, features_hug_batch

def convert_mlm():
    shutil.copyfile(os.path.join(base_path, 'models/hoogberta_base/checkpoint_best.pt'), os.path.join(base_path, 'models/hoogberta_base/model.pt'))
    convert_roberta_checkpoint_to_pytorch('models/hoogberta_base', 'data/converted_model_mlm', False)

    huggingface_model = RobertaForMaskedLM.from_pretrained('data/converted_model_mlm')
    auto_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(base_path, 'data', 'tokenizer.json'), 
        cls_token='<s>', 
        eos_token='</s>', 
        bos_token='<s>', 
        pad_token='<pad>',
        mask_token='<mask>',
        model_max_length = 512
    )

    encoder = HoogBERTaEncoder(cuda=False)

    text = 'เราสามารถพิมภาษาไทยได้ดีมั้ย ทดสอบการทำงาน1223123 😋 ဍဍဍကဏ္ဇ'
    inputText = ["วันที่ 12 มีนาคมนี้","ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"]

    huggingface_model.eval()
    encoder.model.eval()
    
    token_ids_hug, features_hug = feature_extraction_huggingface(
        text, 
        pretokenize, 
        auto_tokenizer, 
        huggingface_model
    )
    token_ids_hug_batch, features_hug_batch = feature_extraction_batch_huggingface(
        inputText, 
        pretokenize, 
        auto_tokenizer, 
        huggingface_model
    )

    token_ids, features = encoder.extract_features(text)
    token_ids_batch, features_batch = encoder.extract_features_batch(inputText)

    with torch.no_grad():
        logit_diff = torch.abs(features - features_hug[0]).mean()
        print('logit_diff:', logit_diff)
        assert logit_diff < 1e-5

    with torch.no_grad():
        mask = token_ids_hug_batch.attention_mask == 1
        logit_diff = torch.abs(features_hug_batch[mask] - features_batch[mask]).mean()
        print('logit_diff batch:', logit_diff)
        assert logit_diff < 1e-5

    assert token_ids_batch.tolist() == token_ids_hug_batch.input_ids.tolist()
    assert token_ids.tolist() == token_ids_hug.input_ids.tolist()[0]

    huggingface_model.save_pretrained('data/converted_model_mlm_huggingface')
    auto_tokenizer.save_pretrained('data/converted_model_mlm_huggingface')

FC_NAME_MAPPER = {
    'POS': 'fc_pos',
    'NER': 'fc_ne',
    'SENTENCE': 'fc_sent'
}

DICT_NAME_MAPPER = {
    'POS': 'pos_dict',
    'NER': 'ne_dict',
    'SENTENCE': 'sent_dict'
}

def predict_and_save(huggingface_model_classification, tagger, sentence, auto_tokenizer, task, prediction_results_original):
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)
    tokenized_text = auto_tokenizer(sentence, return_tensors = 'pt')

    huggingface_model_classification.classifier = getattr(tagger.model, FC_NAME_MAPPER[task])
    id2label = {}
    label2id = {}
    symbols = getattr(tagger, DICT_NAME_MAPPER[task]).symbols
    for i in range(len(symbols)):
        id2label[i] = symbols[i]
        label2id[symbols[i]] = i
    huggingface_model_classification.config.id2label = id2label
    huggingface_model_classification.config.label2id = label2id

    huggingface_model_classification.eval()
    tagger.model.eval()

    with torch.no_grad():
        pred = huggingface_model_classification(**tokenized_text).logits
        out  =  pred.argmax(dim = -1).view(-1).tolist()
        out_result = [symbols[id] for id in out]
    
    assert prediction_results_original[task] == out_result[1:-1]

    huggingface_model_classification.save_pretrained(f"data/converted_model_{task.lower()}_huggingface")
    auto_tokenizer.save_pretrained(f"data/converted_model_{task.lower()}_huggingface")

def convert_multitagger():
    tagger = HoogBERTaMuliTaskTagger(cuda=False) # or cuda=True
    sentence = "วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"
    output_original = tagger.nlp(sentence)

    hoogberta_base_fairseq_checkpoint = torch.load(
        os.path.join(base_path, 'models/hoogberta_base/checkpoint_best.pt'), 
        map_location=torch.device('cpu')
    )
    hoogberta_base_fairseq_checkpoint['model'] = tagger.model.encoder.model.state_dict()
    torch.save(hoogberta_base_fairseq_checkpoint, os.path.join(base_path, 'models/hoogberta_base/model.pt'))
    convert_roberta_checkpoint_to_pytorch('models/hoogberta_base', 'data/converted_model_l12', False)

    huggingface_model_classification = RobertaForTokenClassification.from_pretrained('data/converted_model_l12')
    auto_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(base_path, 'data', 'tokenizer.json'), 
        cls_token='<s>', 
        eos_token='</s>', 
        bos_token='<s>', 
        pad_token='<pad>',
        mask_token='<mask>',
        model_max_length = 512
    )
    
    prediction_results_original = {
        'POS': [],
        'NER': [],
        'SENTENCE': []
    }
    for item in output_original:
        prediction_results_original['POS'].append(item[1])
        prediction_results_original['NER'].append(item[2])
        prediction_results_original['SENTENCE'].append(item[3])

    for task in ['POS', 'NER', 'SENTENCE']:
        predict_and_save(huggingface_model_classification, tagger, sentence, auto_tokenizer, task, prediction_results_original)

if __name__ == '__main__':
    convert_mlm()
    convert_multitagger()