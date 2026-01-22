from .supervised_learning import SupervisedLearning
from .rotnet import RotNet
from .simclr import SimCLR

_METHODS = {
    "supervised_learning": SupervisedLearning,
    "rotnet": RotNet,
    "simclr": SimCLR,
}

def get_method(name: str):
    try:
        return _METHODS[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown method: {name}. Available: {list(_METHODS)}") from e