from .trainer.models import MultiTaskTagger
from .trainer.utils import load_dictionaries
from .trainer.tasks.multitask_tagging import MultiTaskTaggingModule
from attacut import tokenize


class Config(object):

    def __init__(self,layer=12):
        self.feature_layer = layer
        self.model = f"./models/L{layer}/model.ckpt"
        self.task = "multitask-tagging"
        self.traindata = "./"
        self.pretrained = "lst"
        self.dropout = 0.1
        self.do = "inference"


class HoogBERTaMuliTaskTagger(object):

    def __init__(self,layer=12):
        args = Config()
        self.pos_dict, self.ne_dict, self.sent_dict = load_dictionaries(args.traindata)
        self.model = MultiTaskTagger(args,[len(self.pos_dict), len(self.ne_dict), len(self.sent_dict)])
        self.plmodel = MultiTaskTaggingModule(self.model)
        self.plmodel.load_from_checkpoint(checkpoint_path=args.model,model=self.model,strict=False)
        self.srcDict = self.plmodel.model.bert.task.source_dictionary

    def extract_features(self,sentence):
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        input = self.plmodel.model.bert.encode(sentence).unsqueeze(0)
        

    def nlp(self,sentence):
        
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        input = self.plmodel.model.bert.encode(sentence).unsqueeze(0)

        self.plmodel.model(input)

        ppos , pne, pmark = self.plmodel.model(input)
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