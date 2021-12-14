import argparse

class Task(object):

    def __init__(self,args):
        pass

    def setup_task(self,args, parser):
        return None
    
    def train(self,args):
        pass

    def evaluate(self,args, dataset="valid"):
        pass

    