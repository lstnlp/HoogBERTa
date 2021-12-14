TASK_LIST = {}

def register_task(name):

    def register_task_cls(cls):
        #print(cls)
        TASK_LIST[name] = cls
        return cls

    return register_task_cls

def build_task(args,parser,name):
    if name in TASK_LIST:
        TASK_LIST[name].add_args(parser)

        args, _ = parser.parse_known_args()

        task = TASK_LIST[name](args)
        task.setup_task(args,parser)

        return task
    else:
        raise ValueError(f"{name} is not in TASK_LIST.")

from .task import Task
from .multitask_tagging import MultiTaskTagging
