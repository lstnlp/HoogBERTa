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


class HoogBERTaEncoder(object):

    def __init__(self,layer=12):
        args = Config()
        self.pos_dict, self.ne_dict, self.sent_dict = load_dictionaries(args.traindata)
        self.model = MultiTaskTagger(args,[len(self.pos_dict), len(self.ne_dict), len(self.sent_dict)])
        

    def extract_features(self,sentence):
        all_sent = []
        sentences = sentence.split(" ")
        for sent in sentences:
            all_sent.append(" ".join(tokenize(sent)).replace("_","[!und:]"))
        
        sentence = " _ ".join(all_sent)
        
        tokens = self.model.bert.encode(sentence).unsqueeze(0)
        all_layers = self.model.bert.extract_features(tokens, return_all_hiddens=True)
        return tokens[0], all_layers[-1][0]

        

