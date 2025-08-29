"""
Config for training beta vae's on robomimic
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.sailor_vae import SailorVAE
from openx.data.datasets.bridge import bridge_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder, SailorEncoderRNN, SailorDecoderRNN, SailorPrior, ExpandTimeDim
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="16,64"):
    (z_dim, batch_size) = config_str.split(",")
    z_dim = int(z_dim)
    batch_size = int(batch_size)
    seed = 1

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_EULER: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
            "image": {
                "agent": (224, 224),
            }
        },
        "action": {
            "achieved_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
    }

    dataloader = dict(
        datasets=dict(
            bridge=dict(
                path="PATH_TO/bridge/1.0.0",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
            ),
        ),
        global_dataset_statistics="bridge",
        n_obs=1,
        n_action=10,
        shuffle_size=100000,
        batch_size=batch_size,
        recompute_statistics=True, 
        cache=True, 
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
        goal_conditioning="sailor",
        sailor_offset=50,
        chunk_obs=True,
    )

    alg = ModuleSpec.create(
        SailorVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
                "action": None
            },
            trunk=ModuleSpec.create(
                SailorEncoderRNN, mlp=ModuleSpec.create(MLP, [300, 400], activate_final=False)
            ),
        ),
        decoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
                "z": ModuleSpec.create(ExpandTimeDim, 10),
            },
            trunk=ModuleSpec.create(
                SailorDecoderRNN, mlp=ModuleSpec.create(MLP, [300, 400], activate_final=False) # action-dim
            ),
        ),
        prior=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(
                SailorPrior, mlp=ModuleSpec.create(MLP, [1024, 1024], activate_final=False)
            ),
        ),
        tc=ModuleSpec.create(MLP, [128, 128, 1], activate_final=False),
        z_dim=z_dim,
        beta=0.0001,
        tc_weight= 0.000001
    )

    lr_schedule = ModuleSpec.create(optax.constant_schedule, 0.0001)
    optimizer = ModuleSpec.create(optax.adam)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=400000,
            log_freq=500,
            val_freq=2500,
            save_freq=200000,
            val_steps=25,
            seed=seed,
        )
    )
