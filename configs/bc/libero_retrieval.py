# Define the config for robomimic
import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.libero import LIBERO_TASK_IDS, libero_dataset_transform
from openx.data.filters import retrieval_filter
from openx.data.utils import NormalizationType, StateEncoding
from openx.envs.libero_env import LiberoEnv
from openx.networks.action_heads.ddpm import DDPMActionHead
from openx.networks.components.mlp import MLP, LearnedTaskEmbedding
from openx.networks.components.resnet import ResNet18
from openx.networks.components.unet import ConditionalUnet1D
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec

TASK_MAP = {
    "stove_moka": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "bowl_cabinet": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "soup_cheese": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "mug_mug": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",  # noqa: E501
    "book_caddy": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "mug_microwave": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "pots_stove": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "soup_sauce": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "cream_cheese": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "mug_pudding": "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",  # noqa: E501
}


def get_config(config_str: str = "bowl_cabinet,0.5,None,0"):
    # Parse the config string -- used for sweeping.
    task, percent, retrieval, seed = config_str.split(",")
    assert task in TASK_MAP
    percent = float(percent)
    seed = int(seed)

    assert task in TASK_MAP

    datasets = {
        task: dict(
            path="PATH_TO/libero_5_demos/{task}".format(task=task),
            train_split="train",
            transform=ModuleSpec.create(libero_dataset_transform),
            weight=0.5,
        )
    }

    if retrieval.lower() not in {"none", ""}:
        datasets["libero_90"] = dict(
            path="PATH_TO/libero_rlds/libero_90/1.0.0/",
            train_split="train",
            transform=ModuleSpec.create(libero_dataset_transform),
            weight=0.5,
            train_step_filter=ModuleSpec.create(
                retrieval_filter,
                "retrieval/{task}/{retrieval}.pkl".format(task=task, retrieval=retrieval),
                100 - percent,
            ),
        )
        global_dataset_statistics = ["libero_90"]  # Use just the libero90 stats.
    else:
        global_dataset_statistics = [task]

    # Define the structure
    structure = {
        "observation": {
            "state": {
                StateEncoding.EE_POS: NormalizationType.NONE,
                StateEncoding.EE_EULER: NormalizationType.NONE,
                StateEncoding.GRIPPER: NormalizationType.NONE,
            },
            "image": {
                "agent": (128, 128),
                "wrist": (128, 128),
            },
            "task_id": None,
        },
        "action": {
            "desired_delta": {
                StateEncoding.EE_POS: NormalizationType.BOUNDS,
                StateEncoding.EE_EULER: NormalizationType.BOUNDS,
            },
            "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
        },
        "step_idx": None,
        "ep_idx": None,
    }

    dataloader = dict(
        datasets=datasets,
        global_dataset_statistics=global_dataset_statistics,
        n_obs=2,
        n_action=16,
        augment_kwargs=dict(scale_range=(0.85, 1.0), aspect_ratio_range=None),
        shuffle_size=100000,
        batch_size=256,
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
                "observation->image->wrist": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state": None,
                "observation->task_id": ModuleSpec.create(LearnedTaskEmbedding, len(LIBERO_TASK_IDS), 32)
            },
            trunk=ModuleSpec.create(
                Concatenate, model=ModuleSpec.create(MLP, [256], activate_final=False), flatten_time=True
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
            action_horizon=16,
            num_noise_samples=2,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=0.0001,
        warmup_steps=1000,
        decay_steps=200000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    envs = {task: ModuleSpec.create(LiberoEnv, benchmark="libero_10", task=TASK_MAP[task], horizon=500)}
    return ConfigDict(
        dict(
            structure=structure,
            envs=envs,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            # Add training parameters
            steps=100000,
            log_freq=500,
            val_freq=5000,
            eval_freq=25000,
            save_freq=25000,
            val_steps=25,
            n_eval_proc=50,
            eval_ep=50,
            exec_horizon=8,
            seed=seed,
        )
    )
