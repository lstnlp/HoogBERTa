import argparse

from transformers import RobertaForTokenClassification, RobertaForMaskedLM, AutoTokenizer
from huggingface_hub import login

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', help='Huggingface token')
    parser.add_argument('--username', help='Huggingface token')
    args = parser.parse_args()

    login(args.token)

    model_mlm = RobertaForMaskedLM.from_pretrained('data/converted_model_mlm_huggingface')
    model_ner = RobertaForTokenClassification.from_pretrained('data/converted_model_ner_huggingface')
    model_pos = RobertaForTokenClassification.from_pretrained('data/converted_model_pos_huggingface')
    model_sentence = RobertaForTokenClassification.from_pretrained('data/converted_model_sentence_huggingface')

    tokenizer = AutoTokenizer.from_pretrained('data/converted_model_mlm_huggingface')

    model_mlm.push_to_hub(f"{args.username}/HoogBERTa")
    model_ner.push_to_hub(f"{args.username}/HoogBERTa-NER-lst20")
    model_pos.push_to_hub(f"{args.username}/HoogBERTa-POS-lst20")
    model_sentence.push_to_hub(f"{args.username}/HoogBERTa-SENTENCE-lst20")

    tokenizer.push_to_hub(f"{args.username}/HoogBERTa")
    tokenizer.push_to_hub(f"{args.username}/HoogBERTa-NER-lst20")
    tokenizer.push_to_hub(f"{args.username}/HoogBERTa-POS-lst20")
    tokenizer.push_to_hub(f"{args.username}/HoogBERTa-SENTENCE-lst20")