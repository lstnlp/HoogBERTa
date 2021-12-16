from .dataloader import *
from .metrics import *

class Config(object):

    def __init__(self,layer=12,base_path=""):
        self.feature_layer = layer
        self.model = f"./models/L{layer}/model.ckpt"
        self.task = "multitask-tagging"
        self.traindata = "./"
        self.pretrained = "lst"
        self.dropout = 0.1
        self.do = "inference"
        self.base_path=base_path