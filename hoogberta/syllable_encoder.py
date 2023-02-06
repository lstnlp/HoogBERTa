from fairseq.data.dictionary import Dictionary
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from hoogberta.syllacut import tokenize

class HoogBERTaSyllableEncoder(object):

    def __init__(self,layer=12,cuda=False,base_path="."):
        self.base_path = base_path
        roberta = RobertaModel.from_pretrained(self.base_path + '/models/', checkpoint_file='checkpoint.pt',data_name_or_path=self.base_path+"/models/")
        self.roberta = roberta
        self.roberta.eval()
        #print(self.roberta.model.encoder.sentence_encoder)
        #print(self.roberta.task.dictionary)

        if cuda == True:
            self.roberta = self.roberta.cuda()

    
    def extract_features(self,sentence):
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(tokenize(sent).replace("_","[!und:]"))
        
        sentence = "<s> " + " _ ".join(all_sent) #Add BOS
        #print(sentence)

        encode = self.roberta.task.dictionary.encode_line(sentence)
        #print(encode)
        
        tokens = encode.unsqueeze(0)

        output = self.roberta.task.dictionary.string(encode)
        #print("DECODE",output)
        all_layers = self.roberta.extract_features(tokens, return_all_hiddens=True)
        return tokens[0], all_layers[-1][0]

    def extract_features_batch(self,sentenceL):
        
        inputList = []
        for sentX in sentenceL:
            sentences = sentX.split(" ")
            all_sent = []
            for sent in sentences:
                all_sent.append(tokenize(sent).replace("_","[!und:]"))
            
            sentence = " _ ".join(all_sent)
            inputList.append(sentence)

        batch = collate_tokens([self.model.bert.encode(sent) for sent in inputList], pad_idx=1)
        
        #tokens = self.model.bert.encode(inputList)
        return self.extract_features_from_tensor(batch)

    def extract_features_from_tensor(self,batch):
        all_layers = self.model.bert.extract_features(batch, return_all_hiddens=True)
        return batch, all_layers[-1]