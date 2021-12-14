
import sys 
from pathlib import Path
import random

def generate_sentence_boundary_dataset(path='./raw_data/lst20-sent'):

    for subset in ["train", "eval" ,"test"]:
        """ Generate consecutive sentence dataset """
        ftrainS = open(Path.cwd() / path / f"sent.{subset}.th" ,"r").readlines()
        ftrainL = open(Path.cwd() / path / f"sent.{subset}.label" ,"r").readlines()

        otrainS = open(Path.cwd() / path / f"sent1.{subset}.th", "w")
        otrainL = open(Path.cwd() / path / f"sent1.{subset}.label", "w")

        for i in range(0,len(ftrainS),2):
            new_src = ftrainS[i].strip() + " _ " + ftrainS[i+1].strip()
            new_lab = ftrainL[i].strip() + " MARK " + ftrainL[i+1].strip()

            otrainS.writelines(new_src + "\n")
            otrainL.writelines(new_lab + "\n")

            if subset != "test":

                otrainS.writelines(new_src + "\n")
                otrainL.writelines(new_lab + "\n")
        
        otrainS.close()
        otrainL.close()

def generate_shuffle_sentence_boundary_dataset(path='./raw_data/lst20-sent'):

    for subset in ["train", "eval" ,"test"]:
        """ Generate consecutive sentence dataset """
        ftrainS = open(Path.cwd() / path / f"sent.{subset}.th" ,"r").readlines()
        ftrainL = open(Path.cwd() / path / f"sent.{subset}.label" ,"r").readlines()

        dataset = list(zip(ftrainS,ftrainL))
        
        random.shuffle(dataset)

        otrainS = open(Path.cwd() / path / f"sent2.{subset}.th", "w")
        otrainL = open(Path.cwd() / path / f"sent2.{subset}.label", "w")

        for i in range(0,len(dataset),2):
            new_src = dataset[i][0].strip() + " _ " + dataset[i+1][0].strip()
            new_lab = dataset[i][1].strip() + " MARK " + dataset[i+1][1].strip()

            otrainS.writelines(new_src + "\n")
            otrainL.writelines(new_lab + "\n")

            if subset != "test":
                otrainS.writelines(new_src + "\n")
                otrainL.writelines(new_lab + "\n")
        
        otrainS.close()
        otrainL.close()


if __name__ == "__main__":
    generate_sentence_boundary_dataset()
    #generate_shuffle_sentence_boundary_dataset()