"""
Config for training beta vae's on robomimic
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.sailor_vae import SailorVAE
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.core import (
    ExpandTimeDim,
    MultiEncoder,
    SailorDecoderRNN,
    SailorEncoderRNN,
    SailorPrior,
)
from openx.utils.spec import ModuleSpec


def get_config(config_str="16"):
    (z_dim,) = config_str.split(",")
    z_dim = int(z_dim)
    seed = 1

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_QUAT: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.NONE,
                StateEncoding.MISC: NormalizationType.GAUSSIAN,
            },
            "image": {
                "agent": (84, 84),
            },
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
            square_400_paired=dict(
                path="PATH_TO/max_square_400_paired/1.0.0",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(robomimic_dataset_transform),
            ),
        ),
        global_dataset_statistics="square_400_paired",
        n_obs=1,
        n_action=10,
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,  # Small, just recompute.
        cache=True,  # Small enough to stay in memory
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
                "action": None,
            },
            trunk=ModuleSpec.create(SailorEncoderRNN, mlp=ModuleSpec.create(MLP, [300, 400], activate_final=False)),
        ),
        decoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
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
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
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
            val_freq=2500,
            save_freq=100000,
            val_steps=25,
            seed=seed,
        )
    )
