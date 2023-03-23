from fairseq.data.dictionary import Dictionary
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from torch import nn

from hoogberta.syllacut import tokenize

class HoogBERTaSyllableEncoder(nn.Module):

    def __init__(self,layer=12,cuda=False,base_path="."):
        super().__init__()

        self.base_path = base_path
        roberta = RobertaModel.from_pretrained(self.base_path + '/models/', checkpoint_file='checkpoint.pt',data_name_or_path=self.base_path+"/models/",bpe="subword_nmt", bpe_codes=self.base_path + "/models/th_18M.50000.bpe")
        self.roberta = roberta
        self.roberta.eval()
        self.dictionary = self.roberta.task.dictionary
        #print(self.roberta.model.encoder.sentence_encoder)
        #print(self.roberta.task.dictionary)

        #print(type(self.roberta)) :: RobertaHubInterface

        if cuda == True:
            self.roberta = self.roberta.cuda()

    def encode_line(self,line):
        all_sent = []
        sentences = line.split(" ")
        for sent in sentences:
            s = tokenize(sent)
            #print(s)
            all_sent.append(s.replace("_","[!und:]"))

        #Add <s>, </s> will be automatically added
        sentence = "<s> " + " _ ".join(all_sent) 
        encode = self.dictionary.encode_line(sentence,add_if_not_exist=False)
        return encode

    def extract_features(self,sentence):
        encode = self.encode_line(sentence)
        #print(encode)
        out = self.string(encode)
        print(out)
        tokens = encode.unsqueeze(0)
        output = self.roberta.task.dictionary.string(encode)
        #print("DECODE",output)
        all_layers = self.roberta.extract_features(tokens, return_all_hiddens=True)
        return tokens[0], all_layers[-1][0]

    def string(self,tokens):
        return self.dictionary.string(tokens)


    def extract_features_batch(self,sentenceL):
        batch = collate_tokens([self.encode_line(sent) for sent in sentenceL], pad_idx=1)
        
        #tokens = self.model.bert.encode(inputList)
        return self.extract_features_from_tensor(batch)

    def extract_features_from_tensor(self,batch):
        all_layers = self.roberta.extract_features(batch, return_all_hiddens=True)
        return batch, all_layers[-1]