from typing import List

from transformers.models.roberta.convert_roberta_original_pytorch_checkpoint_to_pytorch import convert_roberta_checkpoint_to_pytorch
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM, RobertaForTokenClassification
import torch
from attacut import tokenize as pretokenize

from hoogberta.encoder import HoogBERTaEncoder
from hoogberta.multitagger import HoogBERTaMuliTaskTagger

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
    convert_roberta_checkpoint_to_pytorch('models/hoogberta_base', 'converted_model_mlm', False)

    huggingface_model = RobertaForMaskedLM.from_pretrained('converted_model_mlm')
    auto_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file='tokenizer.json', 
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
    
    token_ids_hug, features_hug = feature_extraction_huggingface(text, pretokenize, auto_tokenizer, huggingface_model)
    token_ids_hug_batch, features_hug_batch = feature_extraction_batch_huggingface(inputText, pretokenize, auto_tokenizer, huggingface_model)

    token_ids, features = encoder.extract_features(text)
    token_ids_batch, features_batch = encoder.extract_features_batch(inputText)

    with torch.no_grad():
        print(torch.abs(features - features_hug[0]).mean())

    with torch.no_grad():
        mask = token_ids_hug_batch.attention_mask == 1
        print(torch.abs(features_hug_batch[mask] - features_batch[mask]).mean())

    print(token_ids_batch == token_ids_hug_batch.input_ids)
    print(token_ids == token_ids_hug.input_ids)


def convert_multitagger():
    tagger = HoogBERTaMuliTaskTagger(cuda=False) # or cuda=True
    output_original = tagger.nlp("วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ")

    hoogberta_base_fairseq_checkpoint = torch.load('/kaggle/working/hoogberta/models/hoogberta_base/model.pt', map_location=torch.device('cpu'))
    hoogberta_base_fairseq_checkpoint['model'] = tagger.model.encoder.model.state_dict()
    torch.save(hoogberta_base_fairseq_checkpoint, '/kaggle/working/hoogberta/models/hoogberta_base/model.pt')
    convert_roberta_checkpoint_to_pytorch('models//hoogberta_base', 'converted_model_l12', False)


    huggingface_model_classification = RobertaForTokenClassification.from_pretrained('converted_model_l12')
    auto_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file='tokenizer.json', 
        cls_token='<s>', 
        eos_token='</s>', 
        bos_token='<s>', 
        pad_token='<pad>',
        mask_token='<mask>',
        model_max_length = 512
    )

    huggingface_model_classification.classifier = tagger.model.fc_ne
    id2label = {}
    label2id = {}
    for i in range(len(tagger.ne_dict.symbols)):
        id2label[i] = tagger.ne_dict.symbols[i]
        label2id[tagger.ne_dict.symbols[i]] = i
    huggingface_model_classification.config.id2label = id2label
    huggingface_model_classification.config.label2id = label2id

    sentence = "วันที่ 12 มีนาคมนี้ ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"

    huggingface_model_classification.eval()
    tagger.model.eval()
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)
    tokenized_text = auto_tokenizer(sentence, return_tensors = 'pt')

    with torch.no_grad():
        ne_pred = huggingface_model_classification(**tokenized_text).logits
        ne_out  =  ne_pred.argmax(dim = -1).view(-1).tolist()
        ne   = [tagger.ne_dict[id] for id in ne_out]