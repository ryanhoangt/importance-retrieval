"""
Config for training beta vae's on robomimic
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.sailor_vae import SailorVAE
from openx.data.datasets.libero import libero_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.components.transformer import ActionEncoder
from openx.networks.core import (
    ExpandTimeDim,
    MultiEncoder,
    SailorDecoderRNN,
    SailorEncoderRNN,
    SailorPrior,
)
from openx.utils.spec import ModuleSpec


def get_config(config_str="16,agent,1"):
    z_dim, cams, seed = config_str.split(",")
    z_dim = int(z_dim)
    assert cams in {"agent"}
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.BOUNDS,
            },
            "image": {"agent": (128, 128)},
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.BOUNDS},
        },
    }

    dataloader = dict(
        datasets=dict(
            libero_90=dict(
                path="PATH_TO/libero_90/1.0.0/",
                train_split="train",
                transform=ModuleSpec.create(libero_dataset_transform),
            ),
            libero_10=dict(
                path="PATH_TO/libero_10/1.0.0/",
                val_split="train",
                transform=ModuleSpec.create(libero_dataset_transform),
            ),
        ),
        global_dataset_statistics="libero_90",
        n_obs=1,
        n_action=10,
        shuffle_size=500000,
        batch_size=128,
        recompute_statistics=False,  # Small, just recompute.
        cache=True,
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
        goal_conditioning="sailor",
        sailor_offset=50,
        chunk_obs_with_action=True,
    )

    alg = ModuleSpec.create(
        SailorVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->state": ModuleSpec.create(MLP, [512], activate_final=True),
                "action": ModuleSpec.create(ActionEncoder, embed_dim=256, output_dim=8, num_layers=2, num_heads=4),
            },
            trunk=ModuleSpec.create(SailorEncoderRNN, mlp=ModuleSpec.create(MLP, [300, 400], activate_final=False)),
        ),
        decoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->state": ModuleSpec.create(MLP, [512], activate_final=True),
                "z": ModuleSpec.create(ExpandTimeDim, 10),
            },
            trunk=ModuleSpec.create(
                SailorDecoderRNN,
                mlp=ModuleSpec.create(MLP, [300, 400], activate_final=False),  # action-dim
            ),
        ),
        prior=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(SailorPrior, mlp=ModuleSpec.create(MLP, [1024, 1024], activate_final=False)),
        ),
        tc=ModuleSpec.create(MLP, [128, 128, 1], activate_final=False),
        z_dim=z_dim,
        beta=0.0001,
        tc_weight=0.000001,
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
            steps=200000,
            log_freq=500,
            val_freq=2000,
            save_freq=25000,
            val_steps=40,
            seed=seed,
        )
    )
