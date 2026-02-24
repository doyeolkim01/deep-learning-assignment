from .supervised_learning import SupervisedLearning
from .rotnet import RotNet
from .simclr import SimCLR
from .moco import MoCo
from .byol import BYOL
from .simsiam import SimSiam


_METHODS = {
    "supervised_learning": SupervisedLearning,
    "rotnet": RotNet,
    "simclr": SimCLR,
    "moco_v1": MoCo,
    "moco_v2": MoCo,
    "byol": BYOL,
    "simsiam": SimSiam,
}

def get_method(name: str):
    try:
        return _METHODS[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown method: {name}. Available: {list(_METHODS)}") from e