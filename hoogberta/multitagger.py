from hoogberta.trainer.models import MultiTaskTagger
from hoogberta.trainer.utils import load_dictionaries,Config
from hoogberta.trainer.tasks.multitask_tagging import MultiTaskTaggingModule

from fairseq.data.data_utils import collate_tokens

from attacut import tokenize
import torch

class HoogBERTaMuliTaskTagger(object):

    def __init__(self,layer=12,cuda=False,base_path="."):
        #print(base_path)
        args = Config(base_path=base_path)
        self.cuda = cuda
        self.pos_dict, self.ne_dict, self.sent_dict = load_dictionaries(base_path)
        self.model = MultiTaskTagger(args,[len(self.pos_dict), len(self.ne_dict), len(self.sent_dict)])
        self.model.eval()

        #Save self.model
        
        state = torch.load(base_path + "/models/L12/modelL12.pt",map_location="cpu")
        #Change name in state dict
        #state = self.update_state_dict(state)
        self.model.load_state_dict(state["model_state_dict"])

        if cuda == True:
            self.model = self.model.cuda()

        #torch.save({"model_state_dict": self.model.state_dict()},"modelL12.pt")
        self.srcDict = self.model.bert.task.source_dictionary

    def update_state_dict(self,old_state):
        state2 = dict()

        for param in old_state['model_state_dict']:
            tensor = old_state['model_state_dict'][param]
            param = param.replace(".decoder.",".encoder.")
            state2[param] = tensor

        return state2


    def extract_features(self,sentence):
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        input = self.model.bert.encode(sentence).unsqueeze(0)
        

    def nlp(self,sentence):
        
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        input = self.model.bert.encode(sentence).unsqueeze(0)

        ppos , pne, pmark = self.model(input)
        pos_out =  ppos.argmax(dim = -1).view(-1).tolist()
        ne_out  =  pne.argmax(dim = -1).view(-1).tolist()
        mark_out = pmark.argmax(dim = -1).view(-1).tolist()

        text = [self.srcDict[tokenid] for tokenid in input[0].tolist()]

        pos  = [self.pos_dict[id] for id in pos_out]
        ne   = [self.ne_dict[id] for id in ne_out]
        mark = [self.sent_dict[id] for id in mark_out]

        out = []
        bpe = 0
        #Remove BPE
        for t,p,n,m in zip(text[1:-1],pos[1:-1],ne[1:-1],mark[1:-1]):
            if bpe == 1:
                out[-1][0] += t.replace("_"," ").replace("[!und:]","_")
            else:
                out.append([t.replace("_"," ").replace("[!und:]","_"),p,n,m])
            
            if t.endswith("@@"):   
                bpe = 1 
                out[-1][0] = out[-1][0][0:-2]
            else:
                bpe = 0

        return out