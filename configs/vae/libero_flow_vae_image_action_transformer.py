"""
Config for training beta vae's on robomimic
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.libero import libero_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP, MLPDecoder
from openx.networks.components.resnet import ResNet18, ResNet18Decoder
from openx.networks.components.transformer import ActionDecoder, ActionEncoder
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str="16,all,1,1,1"):
    z_dim, cams, n_obs, n_action, seed = config_str.split(",")
    z_dim = int(z_dim)
    assert cams in {"all", "agent"}
    n_obs = int(n_obs)
    n_action = int(n_action)
    seed = int(seed)

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
                StateEncoding.EE_EULER: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER: NormalizationType.BOUNDS,
            },
            "flow": {"agent": (128, 128), "wrist": (128, 128)} if cams == "all" else {"agent": (128, 128)},
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
                path="PATH_TO/libero_flow/libero_90/1.0.0",
                train_split="train",
                transform=ModuleSpec.create(libero_dataset_transform),
            ),
            libero_10=dict(
                path="PATH_TO/libero_flow/libero_10/1.0.0",
                val_split="train",
                transform=ModuleSpec.create(libero_dataset_transform),
            ),
        ),
        global_dataset_statistics="libero_90",
        n_obs=2,
        obs_history_keys=["state"],
        n_action=1,
        shuffle_size=100000,
        batch_size=256,
        recompute_statistics=False,  # Small, just recompute.
        cache=False,  # To force reshuffle given the dataset is bigger bc of flow.
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->flow->agent": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->flow->wrist": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->state": ModuleSpec.create(MLP, [512], activate_final=True),
                "action": ModuleSpec.create(ActionEncoder, embed_dim=256, output_dim=8, num_layers=2, num_heads=4),
            }
            if cams == "all"
            else {
                "observation->flow->agent": ModuleSpec.create(ResNet18, num_kp=128),
                "observation->state": ModuleSpec.create(MLP, [512], activate_final=True),
                "action": ModuleSpec.create(ActionEncoder, embed_dim=256, output_dim=8, num_layers=2, num_heads=4),
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [1024, 1024], activate_final=True), flatten_time=True
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [1024, 1024], activate_final=True),
            decoders={
                "observation->flow->agent": ModuleSpec.create(ResNet18Decoder),
                "observation->flow->wrist": ModuleSpec.create(ResNet18Decoder),
                "observation->state": ModuleSpec.create(MLPDecoder, [512]),
                "action": ModuleSpec.create(ActionDecoder, embed_dim=256, num_layers=2, num_heads=4),
            }
            if cams == "all"
            else {
                "observation->flow->agent": ModuleSpec.create(ResNet18Decoder),
                "observation->state": ModuleSpec.create(MLPDecoder, [512]),
                "action": ModuleSpec.create(ActionDecoder, embed_dim=256, num_layers=2, num_heads=4),
            },
        ),
        z_dim=z_dim,
        beta=0.0001,
        weights={
            "observation->state": 1.0,
            "observation->flow->agent": 1 / 100,
            "observation->flow->wrist": 1 / 100,
            "action": 1,
        }
        if cams == "all"
        else {
            "observation->state": 1.0,
            "observation->flow->agent": 1 / 100,
            "action": 1,
        },
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
            steps=250000,
            log_freq=500,
            val_freq=2000,
            save_freq=25000,
            val_steps=40,
            seed=seed,
        )
    )
