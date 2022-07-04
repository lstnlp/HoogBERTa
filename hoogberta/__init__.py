verion = "v0.1"

import shutil
import os 
import gdown 

import zipfile

import hoogberta.trainer

__version__ = "0.1.0"

def download(targetdir=None):
    os.mkdir("./models")
    os.mkdir("./models/L12")
    os.mkdir("./models/hoogberta_base")

    gdown.download("https://drive.google.com/uc?id=1xQHDAE8nbFu2wAM6SAXtTjk890JWLUhy")
    gdown.download("https://drive.google.com/uc?id=1bBSWQzzEt99mYd_EY5W-lQKW6L-D8axW")

    shutil.move("modelL12.pt","./models/L12")

    gdown.download("https://drive.google.com/uc?id=1fYtRAyh6d4W9LVCSJiSYKKM_CCPflBc9") 
    gdown.download("https://drive.google.com/uc?id=1ZNxpVHNZbAfdWA-wu7iMSUtcySzQCJam") 
    gdown.download("https://drive.google.com/uc?id=1ct9xaAUkqxbn9X8JgO8yKfBYWT1Dthld")

    shutil.move("dict.txt","./models/hoogberta_base")
    shutil.move("checkpoint_best.pt","./models/hoogberta_base")
    shutil.move("th_18M.50000.bpe","./models/hoogberta_base")
    

    with zipfile.ZipFile("dict.zip", 'r') as zip_ref:
        zip_ref.extractall("./models")

    os.remove("dict.zip")

    if targetdir is not None:
        shutil.move("./models",targetdir)
