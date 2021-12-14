from . import Task
from . import register_task

import argparse

@register_task("tagging")
class SingleTaskTagging(Task):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--datapath',type=str, default="./raw_data", help='datapath')

        return parser

    def __init__(self,args):
        pass

    def setup_task(self,args):
        return None
    
    def train(self,args):
        pass

    def evaluate(self,args, dataset="valid"):
        pass