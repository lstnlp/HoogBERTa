from hoogberta.trainer.utils import build_dataloader
from hoogberta.trainer.models import register_model

import time
import torch.nn as nn
from torch.nn import ModuleList
import torch

from fairseq.data.dictionary import Dictionary
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.), # can be weight
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.), # everything else
}

def init_all(model, init_funcs):
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)

@register_model("multitask-tagger")
class MultiTaskTagger(nn.Module):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout',type=float, default=0.1, help='fc layer dropout')
        parser.add_argument('--outputdim',type=str, default="10,10,10",  help='list of output dim separated by a comma (,)')
        parser.add_argument('--feature-layer', type=int, default=-1,help='select feature layer (default : -1)')


        return parser

    def __init__(self,args, output_dim = None):
        super().__init__()
        self.args = args
        self.base_path = args.base_path
        self.encoder = self.load_pretrained(args.pretrained)
        self.bert = self.encoder

        embedding_dim = self.bert.model.encoder.sentence_encoder.embed_tokens.weight.size()[1]

        self.fc_pos   = nn.Linear(embedding_dim, output_dim[0])
        self.fc_ne    = nn.Linear(embedding_dim, output_dim[1])
        self.fc_sent  = nn.Linear(embedding_dim, output_dim[2])
        self.dropout = nn.Dropout(args.dropout)
        #For backward compatible issue
        self.fc      = nn.Linear(10,10)


    def load_pretrained(self,pretrained="lst"):
        if pretrained == "lst":
            roberta = RobertaModel.from_pretrained(self.base_path + '/models/hoogberta_base/', checkpoint_file='checkpoint_best.pt',bpe="subword_nmt", bpe_codes=self.base_path + "/models/hoogberta_base/th_18M.50000.bpe",data_name_or_path=".:")
            return roberta

        return None

    def forward(self, token_batch):
        """
        token_batch : Tensor of token ids (long Tensor) [batch , seq length]
        """
        
        all_layers = self.bert.extract_features(token_batch, return_all_hiddens=True)
        last_layer = all_layers[self.args.feature_layer]
        #ic("ALL Layer size",all_layers[-1].size())

        embedded = last_layer
        pos_pred  = self.fc_pos(self.dropout(embedded))
        ne_pred   = self.fc_ne(self.dropout(embedded))
        sent_pred = self.fc_sent(self.dropout(embedded))
        
        #predictions = [sent len, batch size, output dim]
        
        return pos_pred, ne_pred, sent_pred



if __name__ == "__main__":
    print("hi")