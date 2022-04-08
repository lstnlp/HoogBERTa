from hoogberta.trainer.models import MultiTaskTagger
from hoogberta.trainer.utils import load_dictionaries,Config
from hoogberta.trainer.tasks.multitask_tagging import MultiTaskTaggingModule
from fairseq.data.data_utils import collate_tokens
from subword_nmt.apply_bpe import BPE

from attacut import tokenize

class HoogBERTaEncoder(object):

    def __init__(self,layer=12,cuda=False,base_path="."):
        args = Config(base_path=base_path)
        self.base_path = base_path
        self.pos_dict, self.ne_dict, self.sent_dict = load_dictionaries(self.base_path)
        self.model = MultiTaskTagger(args,[len(self.pos_dict), len(self.ne_dict), len(self.sent_dict)])
        self.bpe = BPE(open(self.base_path + "/models/hoogberta_base/th_18M.50000.bpe"))
        if cuda == True:
            self.model = self.model.cuda()
    

    def sentence_to_bpe(self,sentence):
        sentence = " ".join(tokenize(sentence)).replace("_","[!und:]")
        return self.bpe.process_line(sentence)


    def extract_features(self,sentence):
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        tokens = self.model.bert.encode(sentence).unsqueeze(0)
        all_layers = self.model.bert.extract_features(tokens, return_all_hiddens=True)
        return tokens[0], all_layers[-1][0]

    def extract_features_batch(self,sentenceL):
        
        inputList = []
        for sentX in sentenceL:
            sentences = sentX.split(" ")
            all_sent = []
            for sent in sentences:
                all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
            
            sentence = " _ ".join(all_sent)
            inputList.append(sentence)

        batch = collate_tokens([self.model.bert.encode(sent) for sent in inputList], pad_idx=1)
        
        #tokens = self.model.bert.encode(inputList)
        return self.extract_features_from_tensor(batch)

    def extract_features_from_tensor(self,batch):
        all_layers = self.model.bert.extract_features(batch, return_all_hiddens=True)
        return batch, all_layers[-1]
