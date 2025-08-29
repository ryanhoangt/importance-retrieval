# Define the config for robomimic
import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.bridge import bridge_dataset_transform
from openx.data.filters import retrieval_filter
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads.ddpm import DDPMActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.components.unet import ConditionalUnet1D
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec


def get_config(config_str: str = "TO_OVERRIDE,1,TO_OVERRIDE"):
    # Parse the config string -- used for sweeping.
    ret_path, retrieval_percent, dataset_name = config_str.split(",")
    retrieval_percent = float(retrieval_percent)
    weight = float(weight)

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
            },
        },
        "action": {
            "achieved_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
        "ep_idx": None,
        "step_idx": None,
    }

    dataloader = dict(
        datasets=dict(
            bridge_prior=dict(
                path="PATH_TO/bridge/1.0.0/",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
                weight=0.5,
                train_step_filter=ModuleSpec.create(
                    retrieval_filter, f"datasets/{ret_path}.pkl", 100 - retrieval_percent
                ),
            ),
            bridge=dict(
                path=f"PATH_TO/{dataset_name}/bridge/1.0.0/",
                train_split="train",
                val_split="val",
                transform=ModuleSpec.create(bridge_dataset_transform),
                weight=0.5, 
            ),
        ),
        global_dataset_statistics="bridge_prior",
        n_obs=2,
        n_action=4,
        shuffle_size=100000,
        batch_size=128,
        recompute_statistics=False,
        cache=True,  # Small enough to stay in memory
        prefetch=tf.data.AUTOTUNE,  # Enable prefetch.
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [128], activate_final=False), flatten_time=True
            ),
        ),
        action_head=ModuleSpec.create(
            DDPMActionHead,
            model=ModuleSpec.create(
                ConditionalUnet1D, down_features=(256, 512, 1024), mid_layers=2, time_features=128, kernel_size=5
            ),
            clip_sample=1.0,
            timesteps=100,
            variance_type="fixed_small",
            action_dim=7,
            action_horizon=4,
            num_noise_samples=1,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=0.0001,
        warmup_steps=1000,
        decay_steps=500000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    envs = None
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=500000,
            log_freq=500,
            val_freq=2500, # 2500
            eval_freq=20000,
            save_freq=250000,
            val_steps=25,
            n_eval_proc=1,
            eval_ep=20,
            exec_horizon=2,
            seed=0,
        )
    )
