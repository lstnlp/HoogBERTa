from .newmm_tokenizer.tokenizer import syllable_tokenize
from textsearch import TextSearch
import os
import re

ts = TextSearch("insensitive", "norm")


def add_exception(fname):
    global ts
    z = {}
    fp = open(fname,"r").readlines()
    for line in fp:
        line = line.strip()
        if line not in z:
            z[line] = 1
            if "|" in line:
                line,rep = line.split("|")
                ts.add(line.replace("^","").replace("฿",""),rep)
            else:
                ts.add(line.replace("^","").replace("฿",""),line)
    return ts 

#print("Adding Exception ...")
fname = os.path.join(os.path.dirname(__file__), "exceptions.txt")
ts = add_exception(fname)

def preprocess(text):
    text = text.replace("_","[:und!]").replace(" ","_")
    return text

def tokenize(text):
    global ts 
    #text = preprocess(text)
    text = ts.replace(text)
    output = "^".join([i[1:] if i.startswith("^") else i for i in syllable_tokenize(text)]).replace("_","[!und:]").replace(" ","_").split("^")
    return postprocess(" ".join(" ".join(output).split()).replace(" ฿ ",""))

def postprocess(text):
    text = text.replace("[!und:]","_")
    return text

#^([0-9]+|[0-9]{1,3}(,[0-9]{3})*)(.[0-9]+)?$
def is_number(data):
    return re.compile("^([0-9]+|[0-9]{1,3}(,[0-9]{3})*)(.[0-9]+)?$").match(data) != None


def load_ne_list():
    ts = TextSearch("insensitive", "norm")
    fname = os.path.join(os.path.dirname(__file__), "LOC.syllable.txt.sorted")
    for line in open(fname,"r").readlines():
        line = line.strip().replace(" ","|")
        ts.add("|"+line+"|","|<loc>"+line+"</loc>|")

    fname = os.path.join(os.path.dirname(__file__), "PER.syllable.txt.sorted")
    for line in open(fname,"r").readlines():
        line = line.strip().replace(" ","|")
        ts.add("|"+line+"|","|<per>"+line+"</per>|")

    fname = os.path.join(os.path.dirname(__file__), "ORG.syllable.txt.sorted")
    for line in open(fname,"r").readlines():
        line = line.strip().replace(" ","|").replace("_"," ")
        ts.add("|"+line+"|","|<org>"+line+"</org>|")

    fname = os.path.join(os.path.dirname(__file__), "EXP.syllable.txt.sorted")
    for line in open(fname,"r").readlines():
        line = line.strip().replace(" ","|").replace("_"," ")
        ts.add("|"+line+"|","|<exp>"+line+"</exp>|")

    return ts 

def load_possible_syllable():
    ts = [TextSearch("insensitive", dict) for i in range(0,40)]
    fname = os.path.join(os.path.dirname(__file__), "NE.syllable.txt.sorted")
    for line in open(fname,"r").readlines():
        line = line.strip()
        #ts[line.count(" ")].add(" " + line + " ")
        ts[0].add(line)
    return ts

ts_ne = load_ne_list()
ts_est = load_possible_syllable()

def tag_ne(text):
    global ts_ne
    text=text.replace(" ","|").replace("_"," ")
    text = ts_ne.replace(text)
    return text

def estimate_ne(text):
    global ts_est
    text=text
    print(text)
    text = ts_est[0].findall(text)
    return text