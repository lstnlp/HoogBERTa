from sklearn import metrics 
from seqeval.metrics import classification_report, f1_score

from seqeval.scheme import IOBES

def clean_ne(ne):
    if not ne[0:2] in ["I_","B_","E_"] and not ne[0] in ["O"]:
        return "O"
    return ne.replace("_","-")

def clean_sent_label(sent):
    if not sent in ["MARK", "PUNC"]:
        return "O"
    return sent

def calcuate_ne_f1(EVAL_TRUE,EVAL_PRED):
    """
        EVAL_TRUE : List[List[str]] , size = (all sentences, sentence len)
        EVAL_PRED : List[List[str]] , size = (all sentences, sentence len)
    """
    return classification_report(EVAL_TRUE, EVAL_PRED, mode='strict', scheme=IOBES)

def get_pos_accuracy(TRUE,PRED, outfile=None):
    acc = metrics.f1_score(TRUE,PRED, average='micro')
    details = metrics.classification_report(TRUE,PRED, digits=3)
    print(details)

    if outfile is not None:
        fp = open(outfile,"w")
        fp.writelines(details + "\n")
        fp.close()
    return acc

def get_sent_accuracy(TRUE,PRED, outfile=None):
    acc = metrics.f1_score(TRUE,PRED, average='micro')
    detailsStr = metrics.classification_report(TRUE,PRED, digits=3)
    print(detailsStr)

    details = metrics.classification_report(TRUE,PRED, digits=3,output_dict = True)

    if outfile is not None:
        fp = open(outfile,"w")
        fp.writelines(detailsStr + "\n")
        fp.close()
    return details["MARK"]["f1-score"]

def get_ne_accuracy(TRUE,PRED, outfile=None):

    acc = f1_score(TRUE, PRED)
    print(acc)

    details = calcuate_ne_f1(TRUE, PRED)
    print(details)

    if outfile is not None:
        fp = open(outfile,"w")
        fp.writelines(details + "\n")
        fp.close()
    return acc



