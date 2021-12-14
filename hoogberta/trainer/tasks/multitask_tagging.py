from . import Task
from . import register_task
from ..models import build_model
from ..utils import build_dataloader, build_data_iterator, load_dictionaries
from pathlib import Path

import os, json
import time
import torch
import torch.nn as nn

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from seqeval.metrics import classification_report, f1_score
from fairseq.data.data_utils import collate_tokens

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ..utils import get_pos_accuracy, get_ne_accuracy 
from ..utils import clean_ne, get_sent_accuracy, clean_sent_label

MAX_POSITION = 500

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res.long()

def mycollate_tokens(pad_index):

    def myfunc(values):
        #ic("myfunc",len(values))

        return [collate_tokens([values[i][0] for i in range(len(values))],pad_idx=pad_index), 
                collate_tokens([values[i][1] for i in range(len(values))],pad_idx=pad_index),
                collate_tokens([values[i][2] for i in range(len(values))],pad_idx=pad_index),
                collate_tokens([values[i][3] for i in range(len(values))],pad_idx=0)]       # masking

    return myfunc

def map_pos_to_bpe(model, batch): #model = roberta model
    src_raw = []
    tag_raw = []
    new_trg = []
    mask = []
    for i,s in enumerate(batch.src):
        sl = len(batch.src[i].split())
        tl = len(batch.trg[i].split())
        assert sl == tl
        trgi     = '<s> ' + batch.trg[i] + ' </s>'
        trgi     = trgi.split()
        bpe_srci = '<s> ' + model.bpe.encode(s) + ' </s>'
        
        src_raw.append(bpe_srci)
        tag_raw.append(batch.trg[i])

        bpe_srci = bpe_srci.split()
        bpe_trgi = ""
        mask_trg = []
        k = 0
        for j in range(0,len(bpe_srci)):
            if not bpe_srci[j].endswith("@@"):
                bpe_trgi += trgi[k] + " "
                mask_trg.append(1)
                k += 1
            else:
                bpe_trgi += trgi[k] + " "
                mask_trg.append(0)

        new_trg.append(bpe_trgi.strip())

        #mask_trg  [0,1] keep (1) output of that position or not (0)
        #for calucating tagging accuracy 
        mask.append(mask_trg)               
    return new_trg, src_raw, tag_raw, mask

#Pytorch Lightning Trainer 

class MultiTaskTaggingModule(pl.LightningModule):

    def __init__(self, model, optimizer=None, criterion=None, traindata = None, validdata = None):
        super().__init__()
        self.automatic_optimization = True

        self.model = model
        self.args = self.model.args


        self.optimizer = optimizer
        self.criterion = criterion
        self.traindata = traindata
        self.validdata = validdata

        #Save Hyperparameters
        #self.save_hyperparameters(self.args)
        #hparams = vars(self.args)

        if self.args.do == "train":
            os.makedirs("./checkpoints/lstfinetune/" + self.args.checkpoint_dir, exist_ok=True)
            fp = open("./checkpoints/lstfinetune/" + self.args.checkpoint_dir + "/config.josn","w")
            fp.writelines(json.dumps(hparams,indent=2))
            fp.close()
            ic("Saved Hyperparameters !")


    def set_srcdict(self,srcdict):
        self.srcdict = srcdict

    def set_labeldict(self,labeldicts):
        self.labeldicts = labeldicts 

    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        return self.traindata 

    def val_dataloader(self):
        return self.validdata

    def training_step(self,train_batch_dict, batch_idx):
        #print(train_batch.keys())
        batch_loss = 0.0

        train_batch = train_batch_dict["pos"]
        if True: #POS
            predictions, _, _ = self.model.forward(train_batch[0])
        
            #Calcuate Loss
            trg = train_batch[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.reshape((-1,)).long()
            loss = self.criterion(predictions, tags)
            batch_loss += loss
            

        train_batch = train_batch_dict["ne"]
        if True: #NE
            _, predictions, _ = self.model.forward(train_batch[0])
        
            #Calcuate Loss
            trg = train_batch[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.reshape((-1,)).long()
            loss = self.criterion(predictions, tags)
            batch_loss += loss

        train_batch = train_batch_dict["s1"]
        if True: #SENT
            _, _ , predictions = self.model.forward(train_batch[0][:,0:MAX_POSITION])
        
            #Calcuate Loss
            trg = train_batch[1][:,0:MAX_POSITION]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.reshape((-1,)).long()
            loss = self.criterion(predictions, tags)
            batch_loss += loss*0.5

        train_batch = train_batch_dict["s2"]
        if True: #SENT
            _, _ , predictions = self.model.forward(train_batch[0][:,0:MAX_POSITION])
        
            #Calcuate Loss
            trg = train_batch[1][:,0:MAX_POSITION]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.reshape((-1,)).long()
            loss = self.criterion(predictions, tags)
            batch_loss += loss*0.5

        self.log('train_loss', batch_loss.item(),prog_bar=True)

        return batch_loss

    def get_pos_batch_loss(self, inputT, labelT, maskT, loss_type="val"): # [batch, len], maskT = BPE mask
        predictions, _, _ = self.model.forward(inputT)
        predictions_ = predictions.argmax(dim = -1, keepdim = True)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = labelT
        loss = self.criterion(predictions, tags.reshape((-1,)).long())
        loss = loss.sum()
        self.log(f'{loss_type}_loss', loss.item())
        return loss, predictions_ #probabilities

    def get_pos_batch_acc(self, inputT, predT, labelT, maskT):
        batch_prediction = predT #[batch, len]
        batch_prediction = batch_prediction.squeeze(-1).cpu()
        batch_prediction = batch_prediction

        actual = labelT.long().cpu()
        
        PRED   = []
        TRUE   = []
        pred   = []
        actual = []
        for m,b,t,i in zip(maskT,batch_prediction,labelT,inputT):
            temp = []
            for x,y in zip(m[1:],b[1:]): #Not include <s>
                if x == 1:
                    temp.append(self.labeldicts["pos"][y.item()])
            PRED.extend(temp[0:-1])
            pred.append(" ".join(temp[0:-1])) #Not include </s>

            temp = []
            for x,l in zip(m[1:],t[1:]): #Not include <s>
                if x == 1: 
                    temp.append(self.labeldicts["pos"][l.item()])
            TRUE.extend(temp[0:-1])
            actual.append(" ".join(temp[0:-1])) #Not include </s>

        predText   = "\n".join(pred)
        actualText = "\n".join(actual)

        return PRED, TRUE, predText, actualText
    
    def get_ne_batch_loss(self, inputT, labelT, maskT, loss_type="val"): # [batch, len], maskT = BPE mask
        _ , predictions, _ = self.model.forward(inputT)
        predictions_ = predictions.argmax(dim = -1, keepdim = True)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = labelT
        loss = self.criterion(predictions, tags.reshape((-1,)).long())
        loss = loss.sum()
        self.log(f'{loss_type}_loss', loss.item())
        return loss, predictions_ #probabilities

    def get_ne_batch_acc(self, inputT, predT, labelT, maskT):
        batch_prediction = predT #[batch, len]
        batch_prediction = batch_prediction.squeeze(-1).cpu()
        batch_prediction = batch_prediction

        actual = labelT.long().cpu()
        
        PRED   = []
        TRUE   = []
        pred   = []
        actual = []
        for m,b,t,i in zip(maskT,batch_prediction,labelT,inputT):
            temp = []
            for x,y in zip(m[1:],b[1:]): #Not include <s>
                if x == 1:
                    temp.append(clean_ne(self.labeldicts["ne"][y.item()]))
            PRED.extend(temp[0:-1])
            pred.append(" ".join(temp[0:-1])) #Not include </s>

            temp = []
            for x,l in zip(m[1:],t[1:]): #Not include <s>
                if x == 1: 
                    temp.append(clean_ne(self.labeldicts["ne"][l.item()]))
            TRUE.extend(temp[0:-1])
            actual.append(" ".join(temp[0:-1])) #Not include </s>

        predText   = "\n".join(pred)
        actualText = "\n".join(actual)

        return PRED, TRUE, predText, actualText
    
    def get_sent_batch_loss(self, inputT, labelT, maskT, loss_type="val"): # [batch, len], maskT = BPE mask
        _ , _ , predictions = self.model.forward(inputT)
        predictions_ = predictions.argmax(dim = -1, keepdim = True)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = labelT
        loss = self.criterion(predictions, tags.reshape((-1,)).long())
        loss = loss.sum()
        self.log(f'{loss_type}_loss', loss.item())
        return loss, predictions_ #probabilities

    def get_sent_batch_acc(self, inputT, predT, labelT, maskT):
        batch_prediction = predT #[batch, len]
        batch_prediction = batch_prediction.squeeze(-1).cpu()
        batch_prediction = batch_prediction

        actual = labelT.long().cpu()
        
        PRED   = []
        TRUE   = []
        pred   = []
        actual = []
        for m,b,t,i in zip(maskT,batch_prediction,labelT,inputT):
            temp = []
            for x,y in zip(m[1:],b[1:]): #Not include <s>
                if x == 1:
                    temp.append(clean_sent_label(self.labeldicts["sent"][y.item()]))
            PRED.extend(temp[0:-1])
            pred.append(" ".join(temp[0:-1])) #Not include </s>

            temp = []
            for x,l in zip(m[1:],t[1:]): #Not include <s>
                if x == 1: 
                    temp.append(clean_sent_label(self.labeldicts["sent"][l.item()]))
            TRUE.extend(temp[0:-1])
            actual.append(" ".join(temp[0:-1])) #Not include </s>

        predText   = "\n".join(pred)
        actualText = "\n".join(actual)

        return PRED, TRUE, predText, actualText

    def validation_step(self, val_batch, batch_idx, valid_idx):
        
        if valid_idx == 0: #POS
            pos_batch = val_batch
            loss = torch.tensor([0.0]).sum()
            inputText = [" ".join(x.split()) for x in self.srcdict.string(pos_batch[0]).replace("<pad>","").split("\n")]
            inputText = "\n".join(inputText)

            if True:
                inputT = pos_batch[0][:,0:MAX_POSITION]
                labelT = pos_batch[1][:,0:MAX_POSITION]
                maskT  = pos_batch[3][:,0:MAX_POSITION]
                loss, predT = self.get_pos_batch_loss(inputT, labelT, maskT, loss_type="pos_val")
                loss_item = loss.item()
                PRED, ACTUAL, predStr, actualStr = self.get_pos_batch_acc(inputT, predT, labelT, maskT)

            return {"task" : valid_idx ,  'loss'       : loss_item, "srctext"    : inputText, 
                    "predText" : predStr, "actualText" : actualStr,
                    "ACTUAL"   : ACTUAL,  "PRED"       : PRED }

        if valid_idx == 1: #NE
            pos_batch = val_batch
            loss = torch.tensor([0.0]).sum()
            inputText = [" ".join(x.split()) for x in self.srcdict.string(pos_batch[0]).replace("<pad>","").split("\n")]
            inputText = "\n".join(inputText)

            if True:
                inputT = pos_batch[0][:,0:MAX_POSITION]
                labelT = pos_batch[1][:,0:MAX_POSITION]
                maskT  = pos_batch[3][:,0:MAX_POSITION]
                loss, predT = self.get_ne_batch_loss(inputT, labelT, maskT, loss_type="ne_val")
                loss_item = loss.item()
                PRED, ACTUAL, predStr, actualStr = self.get_ne_batch_acc(inputT, predT, labelT, maskT)

            return {"task" : valid_idx ,  'loss'       : loss_item, "srctext"    : inputText, 
                    "predText" : predStr, "actualText" : actualStr,
                    "ACTUAL"   : ACTUAL,  "PRED"       : PRED }

        if valid_idx == 2: #S1
            pos_batch = val_batch
            loss = torch.tensor([0.0]).sum()
            inputText = [" ".join(x.split()) for x in self.srcdict.string(pos_batch[0]).replace("<pad>","").split("\n")]
            inputText = "\n".join(inputText)

            if True:
                inputT = pos_batch[0][:,0:MAX_POSITION]
                labelT = pos_batch[1][:,0:MAX_POSITION]
                maskT  = pos_batch[3][:,0:MAX_POSITION]
                loss, predT = self.get_sent_batch_loss(inputT, labelT, maskT, loss_type="s1_val")
                loss_item = loss.item()
                PRED, ACTUAL, predStr, actualStr = self.get_sent_batch_acc(inputT, predT, labelT, maskT)

            return {"task" : valid_idx ,  'loss'       : loss_item, "srctext"    : inputText, 
                    "predText" : predStr, "actualText" : actualStr,
                    "ACTUAL"   : ACTUAL,  "PRED"       : PRED }
        
        if valid_idx == 3: #S2
            pos_batch = val_batch
            loss = torch.tensor([0.0]).sum()
            inputText = [" ".join(x.split()) for x in self.srcdict.string(pos_batch[0]).replace("<pad>","").split("\n")]
            inputText = "\n".join(inputText)

            if True:
                inputT = pos_batch[0][:,0:MAX_POSITION]
                labelT = pos_batch[1][:,0:MAX_POSITION]
                maskT  = pos_batch[3][:,0:MAX_POSITION]
                loss, predT = self.get_sent_batch_loss(inputT, labelT, maskT, loss_type="s2_val")
                loss_item = loss.item()
                PRED, ACTUAL, predStr, actualStr = self.get_sent_batch_acc(inputT, predT, labelT, maskT)

            return {"task" : valid_idx ,  'loss'       : loss_item, "srctext"    : inputText, 
                    "predText" : predStr, "actualText" : actualStr,
                    "ACTUAL"   : ACTUAL,  "PRED"       : PRED }

    def validation_epoch_end(self, val_step_outputs):
    

        SUMACC = 0.0
        val_step_pos_outputs = val_step_outputs[0]

        #fp = open("out.pos.true.txt","w")
        #fo = open("out.pos.pred.txt","w")

        ACTUAL = []
        PRED   = []

        count = 0
        loss_sum = 0.0

        for out in val_step_pos_outputs:
            if out["task"] == 0: #POS
                loss_sum += out["loss"]
                count += 1
                #fp.writelines(out["actualText"] + "\n")
                #fo.writelines(out["predText"] + "\n")

                ACTUAL.extend(out["ACTUAL"])
                PRED.extend(out["PRED"])
            else:
                print("ERROR")
                exit()

            
        #fp.close()
        #fo.close()

        acc = get_pos_accuracy(ACTUAL,PRED)
        print("POS ACC = %0.3f"%acc)
        SUMACC += acc

        time.sleep(3)
        
        #VALID NE
        val_step_ne_outputs = val_step_outputs[1]

        #fp = open("out.ne.true.txt","w")
        #fo = open("out.ne.pred.txt","w")

        ACTUAL = []
        PRED   = []

        for out in val_step_ne_outputs:
            if out["task"] == 1: #NE
                loss_sum += out["loss"]
                count += 1
                #fp.writelines(out["actualText"] + "\n")
                #fo.writelines(out["predText"] + "\n")

                ACTUAL.append(out["ACTUAL"])
                PRED.append(out["PRED"])
            else:
                print("ERROR")
                exit()
            
        #fp.close()
        #fo.close()

        acc = get_ne_accuracy(ACTUAL,PRED)
        print("NE ACC = %0.3f"%acc)
        SUMACC += acc

        time.sleep(3)

        #VALID SENT1
        val_step_ne_outputs = val_step_outputs[2]

        #fp = open("out.s1.true.txt","w")
        #fo = open("out.s1.pred.txt","w")

        ACTUAL = []
        PRED   = []

        for out in val_step_ne_outputs:
            if out["task"] == 2: #
                loss_sum += out["loss"]
                count += 1
                #fp.writelines(out["actualText"] + "\n")
                #fo.writelines(out["predText"] + "\n")

                ACTUAL.extend(out["ACTUAL"])
                PRED.extend(out["PRED"])
            else:
                print("ERROR")
                exit()
            
        #fp.close()
        #fo.close()

        acc = get_sent_accuracy(ACTUAL,PRED,outfile="out.s1.acc.txt")
        print("SENT1 ACC = %0.3f"%acc)
        SUMACC += acc

        time.sleep(3)

        #VALID SENT2
        val_step_ne_outputs = val_step_outputs[3]

        #fp = open("out.s2.true.txt","w")
        #fo = open("out.s2.pred.txt","w")

        ACTUAL = []
        PRED   = []

        for out in val_step_ne_outputs:
            if out["task"] == 3: #
                loss_sum += out["loss"]
                count += 1
                #fp.writelines(out["actualText"] + "\n")
                #fo.writelines(out["predText"] + "\n")

                ACTUAL.extend(out["ACTUAL"])
                PRED.extend(out["PRED"])
            else:
                print("ERROR")
                exit()
            
        #fp.close()
        #fo.close()

        acc = get_sent_accuracy(ACTUAL,PRED,outfile="out.s2.acc.txt")
        print("SENT2 ACC = %0.3f"%acc)
        SUMACC += acc

        time.sleep(3)

        self.log("val_loss", loss_sum / count)
        self.log("val_acc",SUMACC)

    def set_test_task(self,task="pos"):
        self.test_task = task 

    def test_step(self,test_batch, batch_idx):

        if self.test_task == "pos":
            pos_batch = test_batch
            inputT = pos_batch[0][:,0:MAX_POSITION]
            labelT = pos_batch[1][:,0:MAX_POSITION]
            maskT  = pos_batch[3][:,0:MAX_POSITION]
            predictions, _, _ = self.model.forward(inputT)
            predictions_ = predictions.argmax(dim = -1, keepdim = True)
            predT = predictions_
            PRED, ACTUAL, predStr, actualStr = self.get_pos_batch_acc(inputT, predT, labelT, maskT)

            return { "task" : 0,
                "predText" : predStr, "actualText" : actualStr,
                "ACTUAL"   : ACTUAL,  "PRED"       : PRED }
        
        if self.test_task == "ne":
            pos_batch = test_batch
            inputT = pos_batch[0][:,0:MAX_POSITION]
            labelT = pos_batch[1][:,0:MAX_POSITION]
            maskT  = pos_batch[3][:,0:MAX_POSITION]
            _ , predictions, _ = self.model.forward(inputT)
            predictions_ = predictions.argmax(dim = -1, keepdim = True)
            predT = predictions_
            PRED, ACTUAL, predStr, actualStr = self.get_ne_batch_acc(inputT, predT, labelT, maskT)

            return { "task" : 0,
                "predText" : predStr, "actualText" : actualStr,
                "ACTUAL"   : ACTUAL,  "PRED"       : PRED }

        if self.test_task == "sent1" or self.test_task == "sent2":
            pos_batch = test_batch
            inputT = pos_batch[0][:,0:MAX_POSITION]
            labelT = pos_batch[1][:,0:MAX_POSITION]
            maskT  = pos_batch[3][:,0:MAX_POSITION]
            _ , _, predictions = self.model.forward(inputT)
            predictions_ = predictions.argmax(dim = -1, keepdim = True)
            predT = predictions_
            PRED, ACTUAL, predStr, actualStr = self.get_sent_batch_acc(inputT, predT, labelT, maskT)

            return { "task" : 0,
                "predText" : predStr, "actualText" : actualStr,
                "ACTUAL"   : ACTUAL,  "PRED"       : PRED }



    def test_epoch_end(self,test_step_outputs):
        acc = 0         
        ACTUAL = []
        PRED   = []
        
        if self.test_task == "pos":
            for out in test_step_outputs:
                ACTUAL.extend(out["ACTUAL"])
                PRED.extend(out["PRED"])
            acc = get_pos_accuracy(ACTUAL,PRED)
            print("POS ACC = %0.3f"%acc)

        if self.test_task == "ne":
            for out in test_step_outputs:
                ACTUAL.append(out["ACTUAL"])
                PRED.append(out["PRED"])
            acc = get_ne_accuracy(ACTUAL,PRED)
            print("NE ACC = %0.3f"%acc)

        if self.test_task == "sent1" or self.test_task == "sent2":
            for out in test_step_outputs:
                ACTUAL.extend(out["ACTUAL"])
                PRED.extend(out["PRED"])
            acc = get_sent_accuracy(ACTUAL,PRED)
            print("SENT ACC = %0.3f"%acc)

        return acc

@register_task("multitask-tagging")
class MultiTaskTagging(Task):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--traindata',type=str, default="./raw_data", help='datapath')
        parser.add_argument('--valid-interval',type=int, default="500", help='validation interval')
        parser.add_argument('--testsubset',type=str, default="test", help='test subset (select from ["eval","test"])')
        parser.add_argument('--sample', action="store_true")
        parser.add_argument('--save-location',type=str, default="./model.pt", help='model output path')
        return parser

    def __init__(self,args):
        self.pad_index = 1
        pass

    def setup_task(self, args, parser):
        self.args = args
        self.loadall = not args.sample

        #Setup Dictionary and Pad Index
        pos_dict, ne_dict, sent_dict = load_dictionaries(args.traindata)
        taskdict = {"pos" : pos_dict , "ne" : ne_dict, "sent" : sent_dict, "sent1" : sent_dict, "sent2" : sent_dict}

        
        self.taskdict = taskdict

        self.pad_idx = pos_dict.pad_index
        outdim = [len(pos_dict.symbols), len(ne_dict.symbols), len(sent_dict.symbols) ]

        #Setup Model
        kwargs = {"output_dim" : outdim}
        model = build_model(parser, args.model, **kwargs)
        self.model = model

        ic()
        ic(model.dropout)

        if args.do == "train":
                

            #Setup Criterion
            criterion = nn.CrossEntropyLoss(ignore_index = self.pad_index)

            #Setup Optimizer
            LEARNING_RATE = args.lr
            optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9)

            #Setup Dataset
            ic("Loading POS dataset ...")
            ic(args.batch_size)
            
            taskdict = {"pos" : pos_dict , "ne" : ne_dict, "sent" : sent_dict, "sent1" : sent_dict, "sent2" : sent_dict}

            #LOAD POS Dataset
            dataset = "pos"

            trainSetpos = build_data_iterator(args,args.traindata,dataset=dataset,type="train")
            trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainPosData = DataLoader(trainSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            validSetpos = build_data_iterator(args,args.traindata,
                                            dataset=dataset,type="eval",shuffle=False)

            validSetPos_tensor = self.convert_to_tensor(validSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            validPosData = DataLoader(validSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            
            #LOAD NE Dataset
            ic("Loading NE dataset ...")
            dataset = "ne"
            trainSetne = build_data_iterator(args,args.traindata,dataset=dataset,type="train")
            trainSetne_tensor = self.convert_to_tensor(trainSetne,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainNeData = DataLoader(trainSetne_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            validSetne = build_data_iterator(args,args.traindata,
                                            dataset=dataset,type="eval",shuffle=False)

            validSetNe_tensor = self.convert_to_tensor(validSetne,
                                                        label_encoder=taskdict[dataset].encode_line)
            validNeData = DataLoader(validSetNe_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            #LOAD SENT1 Dataset
            ic("Loading SENT1 dataset ...")
            dataset = "sent1"
            trainSetS1 = build_data_iterator(args,args.traindata,dataset=dataset,type="train")
            trainSetS1_tensor = self.convert_to_tensor(trainSetS1,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainS1Data = DataLoader(trainSetS1_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            validSetS1 = build_data_iterator(args,args.traindata,
                                            dataset=dataset,type="eval",shuffle=False)

            validSetS1_tensor = self.convert_to_tensor(validSetS1,
                                                        label_encoder=taskdict[dataset].encode_line)
            validS1Data = DataLoader(validSetS1_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    collate_fn = mycollate_tokens(self.pad_idx))
            
            #LOAD SENT1 Dataset
            ic("Loading SENT2 dataset ...")
            dataset = "sent2"
            trainSetS2 = build_data_iterator(args,args.traindata,dataset=dataset,type="train")
            trainSetS2_tensor = self.convert_to_tensor(trainSetS2,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainS2Data = DataLoader(trainSetS2_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            validSetS2 = build_data_iterator(args,args.traindata,
                                            dataset=dataset,type="eval",shuffle=False)

            validSetS2_tensor = self.convert_to_tensor(validSetS2,
                                                        label_encoder=taskdict[dataset].encode_line)
            validS2Data = DataLoader(validSetS2_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    collate_fn = mycollate_tokens(self.pad_idx))
            

            #Setup Trainer
            ic("Loading trainer ...")
            checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/lstfinetune/' + args.checkpoint_dir, monitor = 'val_acc', save_top_k=5, mode='max',every_n_val_epochs=1, filename="{epoch}-{step}-{val_loss:0.5f}-{val_acc:.3f}")

            earlystop_callback = EarlyStopping(monitor='val_acc', patience=5, mode='max', check_on_train_epoch_end=False,verbose=True)

            callbacks = [checkpoint_callback,earlystop_callback]
  
            trainDataLoader = {"pos" : trainPosData, "ne" : trainNeData, "s1" : trainS1Data, "s2" : trainS2Data}
            validDataLoader = [validPosData, validNeData, validS1Data, validS2Data]

            self.plmodel = MultiTaskTaggingModule(model, optimizer,criterion,trainDataLoader,validDataLoader)
            
            self.plmodel.set_srcdict(self.model.bert.task.source_dictionary)
            self.plmodel.set_labeldict(taskdict)
            
            self.trainer = pl.Trainer(gpus=args.gpus, accelerator=args.accelerator,
                    val_check_interval=args.valid_interval, 
                    multiple_trainloader_mode="max_size_cycle",
                    reload_dataloaders_every_n_epochs=1,callbacks=callbacks,
                    resume_from_checkpoint=args.resume)

        elif args.do == "test":
            taskdict = {"pos" : pos_dict , "ne" : ne_dict, "sent" : sent_dict, "sent1" : sent_dict, "sent2" : sent_dict}
            self.plmodel = MultiTaskTaggingModule(model)
            self.plmodel.set_srcdict(self.model.bert.task.source_dictionary)
            self.plmodel.set_labeldict(taskdict)

            self.plmodel.load_from_checkpoint(checkpoint_path=args.resume,model=self.model,strict=False)

            self.trainer = pl.Trainer(gpus=args.gpus, 
                    resume_from_checkpoint=args.resume)

        elif args.do == "save":
            self.plmodel = MultiTaskTaggingModule(model)
            self.plmodel.set_srcdict(self.model.bert.task.source_dictionary)
            self.plmodel.set_labeldict(taskdict)
            self.plmodel.load_from_checkpoint(checkpoint_path=args.resume,model=self.model)
            torch.save(self.model.state_dict(), self.args.save_location)
            ic("Model Saved !")

        elif args.do == "load":
            model.load_state_dict(torch.load(args.save_location))

            self.plmodel = MultiTaskTaggingModule(model)
            self.plmodel.set_srcdict(self.model.bert.task.source_dictionary)
            self.plmodel.set_labeldict(taskdict)
            self.trainer = pl.Trainer(gpus=args.gpus)

        elif args.do == "count":
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            ic("Numbers of parameter : ",params)

        else:
            print("Error args.do should be 'train' or 'test.")
            
        return None

    def convert_to_tensor(self,data, label_encoder):
        """
        Convert String to Tensor
        """
        
        srcBPETensor = []
        trgBPETensor = []
        trgORITensor = []
        mskBPETensor = []

        srcDict = self.model.bert.task.source_dictionary
        for idx,batch in enumerate(data):
            if idx % 100 == 0:
                ic(idx)

            if not self.loadall:
                if idx == 50:
                    break

            trgBPEList, srcBPEList, trgORIList, mskBPEList = map_pos_to_bpe(self.model.bert,batch)
            #ic(mskBPEList)
            
            srcBPETensor.extend([srcDict.encode_line(lineT, append_eos=False, add_if_not_exist=False) for lineT in srcBPEList])

            trgBPETensor.extend([label_encoder(lineT,append_eos=False, add_if_not_exist=False) for lineT in trgBPEList])

            trgORITensor.extend([label_encoder(lineT, append_eos=False, add_if_not_exist=False) for lineT in trgORIList])

            mskBPETensor.extend([torch.tensor(msk,dtype=torch.int32) for msk in mskBPEList])

        return list(zip(srcBPETensor, trgBPETensor, trgORITensor,mskBPETensor))
    
    def train(self):
        self.trainer.fit(self.plmodel)
        return 

    def evaluate(self, subset=None):
        args = self.args

        if subset is None:
            subset = args.testsubset

        self.loadall = not args.sample

        taskdict = self.taskdict
        outdim = [len(taskdict["pos"].symbols), len(taskdict["ne"].symbols), len(taskdict["sent"].symbols) ]

        #Load Dataset
        #LOAD POS Dataset
        if True:
            ic("POS Dataset")
            dataset = "pos"
            self.plmodel.set_test_task("pos")

            trainSetpos = build_data_iterator(args,args.traindata,dataset=dataset,type=subset)
            trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainPosData = DataLoader(trainSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            trainer = self.trainer
            trainer.test(self.plmodel, dataloaders=trainPosData)

        #Test NER
        if True:
            ic("NE Dataset")
            dataset = "ne"
            self.plmodel.set_test_task("ne")

            trainSetpos = build_data_iterator(args,args.traindata,dataset=dataset,type=subset)
            trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainPosData = DataLoader(trainSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            trainer = self.trainer
            trainer.test(self.plmodel, dataloaders=trainPosData)

        if True:
            ic("Sent 1 Dataset")
            dataset = "sent1"
            self.plmodel.set_test_task("sent1")

            trainSetpos = build_data_iterator(args,args.traindata,dataset=dataset,type=subset)
            trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainPosData = DataLoader(trainSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            trainer = self.trainer
            trainer.test(self.plmodel, dataloaders=trainPosData)
        
        if True:
            ic("Sent 2 Dataset")
            dataset = "sent2"
            self.plmodel.set_test_task("sent1")

            trainSetpos = build_data_iterator(args,args.traindata,dataset=dataset,type=subset)
            trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                        label_encoder=taskdict[dataset].encode_line)
            trainPosData = DataLoader(trainSetPos_tensor, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    collate_fn = mycollate_tokens(self.pad_idx))

            trainer = self.trainer
            trainer.test(self.plmodel, dataloaders=trainPosData)

        return None