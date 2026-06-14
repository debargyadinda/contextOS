from .lru import LRUPolicy
from .lfu import LFUPolicy
from .clock import CLOCKPolicy
from .random_policy import RandomPolicy
from .asr import ASRPolicy
from .working_set import WorkingSetPolicy

ALL_POLICIES = [
    LRUPolicy(),
    LFUPolicy(),
    CLOCKPolicy(),
    RandomPolicy(),
    WorkingSetPolicy(),
    ASRPolicy(),
]
