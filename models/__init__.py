from .resnet import resnet20_encoder, resnet32_encoder
from .preactresnet import preactresnet20_encoder, preactresnet32_encoder
from .densenet import densenet_encoder
from .vit import vit_encoder 
from .mlpmixer import mlpmixer_encoder
from .convmixer import convmixer_encoder
from .fractalnet import fractalnet_encoder

_ENCODERS = {
    "resnet20": resnet20_encoder,
    "resnet32": resnet32_encoder,
    "preactresnet20": preactresnet20_encoder,
    "preactresnet32": preactresnet32_encoder,
    "densenet": densenet_encoder,
    "fractalnet": fractalnet_encoder,
    "vit": vit_encoder,
    "mlpmixer": mlpmixer_encoder,
    "convmixer": convmixer_encoder,
    
}

def get_encoder(name: str):
    try:
        return _ENCODERS[name.lower()]()
    except KeyError as e:
        raise ValueError(f"Unknown model: {name}. Available: {list(_ENCODERS)}") from e
