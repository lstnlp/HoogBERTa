MODEL_LIST = {}

def register_model(name):

    def register_model_cls(cls):
        #print(cls)
        MODEL_LIST[name] = cls
        return cls

    return register_model_cls


def build_model(parser,name,**kwargs):
    if name in MODEL_LIST:
        MODEL_LIST[name].add_args(parser)
        args, _ = parser.parse_known_args()
        model = MODEL_LIST[name](args,**kwargs)
        return model
    else:
        raise ValueError(f"{name} is not in MODEL_LIST.")

from .multitask_tagger import MultiTaskTagger