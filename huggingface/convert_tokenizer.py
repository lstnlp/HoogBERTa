import os
from typing import List

from attacut import tokenize as pretokenize
from fairseq.data.data_utils import collate_tokens
from tqdm.auto import tqdm
from tokenizers.implementations.char_level_bpe import CharBPETokenizer
from tokenizers.processors import TemplateProcessing
import torch

from hoogberta.encoder import HoogBERTaEncoder

base_path = os.path.abspath(os.getcwd())

def tokenize_subwordnmt(sentence: str, pretokenize, encoder):
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)

    return encoder.model.encoder.encode(sentence)

def tokenize_batch_subwordnmt(sentenceL: List[str], pretokenize, encoder):

    inputList = []
    for sentX in sentenceL:
        sentences = sentX.split(" ")
        all_sent = []
        for sent in sentences:
            all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

        sentence = " _ ".join(all_sent)
        inputList.append(sentence)

    batch = collate_tokens([encoder.model.encoder.encode(sent) for sent in inputList], pad_idx=1)

    return batch

def tokenize_huggingface_tokenizer(sentence: str, pretokenize, huggingface_bpe):
    all_sent = []
    sentences = sentence.split(" ")
    for sent in sentences:
        all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

    sentence = " _ ".join(all_sent)

    return huggingface_bpe.encode(sentence).ids

def tokenize_batch_huggingface_tokenizer(sentenceL: List[str], pretokenize, huggingface_bpe):

    inputList = []
    for sentX in sentenceL:
        sentences = sentX.split(" ")
        all_sent = []
        for sent in sentences:
            all_sent.append(" ".join(pretokenize(sent)).replace("_","[!und:]"))

        sentence = " _ ".join(all_sent)
        inputList.append(sentence)
    
    batch = collate_tokens([torch.Tensor(huggingface_bpe.encode(sent).ids).long() for sent in inputList], pad_idx=1)

    return batch

if __name__ == '__main__':
    encoder = HoogBERTaEncoder(cuda=False)

    merges = []
    with open(os.path.join(base_path, "models/hoogberta_base/th_18M.50000.bpe")) as f:
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            left, right = line.rstrip().split(' ')
            merges.append((left, right))


    vocabs_hoog = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(os.path.join(base_path, 'models/hoogberta_base/dict.txt')) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            subword = line.rstrip().split(' ')[0]
            if subword[-2:] == '@@':
                subword = subword[:-2]
            else:
                subword += '</w>'
            vocabs_hoog.append(subword)
            
    vocabs_hoog.append('<mask>')

    vocab_converted_hoog = {}
    for i in range(len(vocabs_hoog)):
        vocab_converted_hoog[vocabs_hoog[i]] = i

    """
    Somehow some vocab in merge.txt did not registered in dict.txt file. 
    We can easily fix that by removing not registered index from merge rules.
    """

    i = 0
    error_index = []
    error_coms = []
    for item in merges:
        item_com = item[0] + item[1]
        if item_com not in vocab_converted_hoog or item[0] not in vocab_converted_hoog or item[1] not in vocab_converted_hoog:
            error_index.append(i)
        i += 1

    merges_fixed = merges.copy()
    for i in sorted(error_index, reverse=True):
        del merges_fixed[i]


    huggingface_bpe = CharBPETokenizer(
        merges=merges_fixed, 
        vocab=vocab_converted_hoog, 
        bert_normalizer = False, 
        split_on_whitespace_only = True, 
        unk_token='<unk>'
    )


    huggingface_bpe.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>"])
    huggingface_bpe.add_special_tokens(["<mask>"])
    huggingface_bpe.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )


    text = 'เราสามารถพิมภาษาไทยได้ดีมั้ย ทดสอบการทำงาน1223123 😋 ဍဍဍကဏ္ဇ'
    inputText = ["วันที่ 12 มีนาคมนี้","ฉันจะไปเที่ยววัดพระแก้ว ที่กรุงเทพ"]

    subwordnmt_result = tokenize_subwordnmt(text, pretokenize, encoder)
    huggingface_result = tokenize_huggingface_tokenizer(text, pretokenize, huggingface_bpe)

    subwordnmt_result_batch = tokenize_batch_subwordnmt(inputText, pretokenize, encoder)
    huggingface_result_batch = tokenize_batch_huggingface_tokenizer(inputText, pretokenize, huggingface_bpe)


    assert subwordnmt_result.tolist() == huggingface_result
    assert subwordnmt_result_batch.tolist() == huggingface_result_batch.tolist()


    directory = 'data'
    if not os.path.exists(os.path.join(base_path, directory)):
        os.makedirs(os.path.join(base_path, directory))
    huggingface_bpe.save(os.path.join(directory, 'tokenizer.json'))