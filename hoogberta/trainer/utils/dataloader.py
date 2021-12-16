from pathlib import Path
from torch.utils.data import DataLoader
from fairseq.data.dictionary import Dictionary
from fairseq.data.data_utils import collate_tokens


from torchtext.data import BucketIterator
from torchtext import data
from torchtext import datasets


def build_data_iterator(args,path,dataset="pos",type="train",shuffle=False):
    BATCH_SIZE = args.batch_size
    path = Path.cwd() / path

    src = data.RawField()
    trg = data.RawField()
    
    mt_train = datasets.TranslationDataset(
        path= str(path / f'./lst20-{dataset}/{dataset}.{type}'), exts=('.th', '.label'),
        fields=(src, trg))
    if shuffle:
        train_iter = data.BucketIterator(mt_train,sort=True, sort_within_batch=False, 
                                        batch_size=BATCH_SIZE, device='cpu', shuffle=shuffle,sort_key=lambda x: len(x.src))
    else:
        train_iter = data.BucketIterator(mt_train,sort=False, sort_within_batch=False, 
                                        batch_size=BATCH_SIZE, device='cpu', shuffle=shuffle)
    return iter(train_iter)

def load_dictionaries(path):
    path = Path(path)
    pos_dict = Dictionary().load(open(path / "./models/dict/pos.txt","r",encoding="utf-8"))
    pos_dict.add_symbol("<s>")
    pos_dict.add_symbol("</s>")
    pos_dict.finalize()
    #print(pos_dict.symbols)

    ne_dict = Dictionary().load(open(path / "./models/dict/ne.txt","r",encoding="utf-8"))
    ne_dict.add_symbol("<s>")
    ne_dict.add_symbol("</s>")
    ne_dict.finalize()
    #print(ne_dict.symbols)

    sent_dict = Dictionary().load(open(path / "./models/dict/sent.txt","r",encoding="utf-8"))
    sent_dict.add_symbol("<s>")
    sent_dict.add_symbol("</s>")
    sent_dict.finalize()
    #print(sent_dict.symbols)

    return pos_dict, ne_dict, sent_dict



def build_dataloader(path,dataset="pos",batch_size=32,shuffle=True,collate_tokens=None):
    path = Path.cwd() / path
    sent  = path / f'lst20-{dataset}' / f'{dataset}.train.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.train.label'
    sentTrainList  = [line.strip().split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip().split() for line in open(label,"r").readlines()]
    trainCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    train_dataloader = DataLoader(trainCorpus, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_tokens)

    sent  = path / f'lst20-{dataset}' / f'{dataset}.eval.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.eval.label'
    sentTrainList  = [line.strip().split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip().split() for line in open(label,"r").readlines()]
    evalCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    valid_dataloader = DataLoader(evalCorpus, batch_size=batch_size, shuffle=False, collate_fn=collate_tokens)

    sent  = path / f'lst20-{dataset}' / f'{dataset}.test.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.test.label'
    sentTrainList  = [line.strip().split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip().split() for line in open(label,"r").readlines()]
    testCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    test_dataloader = DataLoader(testCorpus, batch_size=batch_size, shuffle=False,collate_fn=collate_tokens)

    return train_dataloader, valid_dataloader, test_dataloader