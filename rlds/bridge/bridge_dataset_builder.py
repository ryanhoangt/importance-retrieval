import glob
import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from PIL import Image

# we ignore the small amount of data that contains >4 views
N_VIEWS = 4
IMAGE_SIZE = (256, 341)

DEPTH = 0 
N_TRAIN = 10 # rest are used for val

COMPUTE_FLOW = False
if COMPUTE_FLOW:
    import torch
    from gmflow.gmflow.gmflow import GMFlow
    import torch.nn.functional as F


ORIG_NAMES = [f"images{i}" for i in range(N_VIEWS)]
NEW_NAMES = [f"image_{i}" for i in range(N_VIEWS)]

IMAGE_0_TOPICS = [
    "/cam0/image_raw",
    "/camera0/color/image_raw",
    "/D435/color/image_raw",
    "/blue/image_raw",
]

OTHER_TOPCIS = [
    "/cam1/image_raw",
    "/cam2/image_raw",
    "/cam3/image_raw",
    "/cam4/image_raw",
    "/camera1/color/image_raw",
    "/camera3/color/image_raw",
    "/camera2/color/image_raw",
    "/camera4/color/image_raw",
    "/yellow/image_raw",
]


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        # depth should be uint16 (I;16), but PIL has a bug where it reads as int32 (I)
        # there are also few trajectories where it's uint8 (L) for some reason
        # we just cast to uint16 in both cases
        assert im.mode == "RGB"
        assert im.size == (640, 480), (path, im.size)
        arr = np.array(im)
        assert arr.ndim == 3 and arr.shape[-1] == 3, (path, arr.shape)
        assert arr.dtype == np.uint8, (path, arr.dtype)
        arr = tf.image.resize(arr, IMAGE_SIZE, method="lanczos3", antialias=True)
        arr = tf.cast(tf.clip_by_value(tf.round(arr), 0, tf.dtypes.uint8.max), tf.uint8)
        return arr._numpy()


def read_depth(path: str):
    with Image.open(path) as im:
        assert im.mode == "I" or im.mode == "L", (path, im.mode)
        assert im.size == (640, 480), (path, im.size)
        arr = np.array(im)
        arr = arr[..., None].astype(np.uint16)
        arr = tf.image.resize(arr, IMAGE_SIZE, method="bilinear", antialias=True)
        arr = tf.cast(tf.clip_by_value(tf.round(arr), 0, tf.dtypes.uint16.max), tf.uint16)
        return arr._numpy()


def process_images(path):  # processes images at a trajectory level
    image_dirs = set(os.listdir(str(path))).intersection(set(ORIG_NAMES))
    image_paths = [
        sorted(
            glob.glob(os.path.join(path, image_dir, "im_*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for image_dir in image_dirs
    ]

    filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
    assert all(x == filenames[0] for x in filenames), (path, filenames)

    return {image_dir: [read_image(path) for path in p] for image_dir, p in zip(image_dirs, image_paths, strict=False)}


def process_depth(path):
    depth_path = os.path.join(path, "depth_images0")
    if os.path.exists(depth_path):
        image_paths = sorted(
            glob.glob(os.path.join(depth_path, "im_*.png")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        return [read_depth(path) for path in image_paths]
    return None


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"]


def process_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def process_lang(path):
    fp = os.path.join(path, "lang.txt")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()

    return text


class Bridge(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bridge dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "You can download the raw BridgeData from https://rail.eecs.berkeley.edu/datasets/bridge_release/data/."
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        if COMPUTE_FLOW:
            features = tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_0_flow": tfds.features.Tensor(
                                        shape=(*IMAGE_SIZE, 2),
                                        dtype=np.float32,
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "ep_idx": tfds.features.Scalar(dtype=np.int32, doc="Deterministic index of the episode."),
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        else:
            features = tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=(*IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "ep_idx": tfds.features.Scalar(dtype=np.int32, doc="Deterministic index of the episode."),
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        return self.dataset_info_from_configs(
            features=features
        )

    @classmethod
    def _process_example(cls, ep_idx, example_input):
        """Process a single example."""
        path, camera_topics = example_input

        if COMPUTE_FLOW:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GMFlow(feature_channels=128,
                            num_scales=1,
                            upsample_factor=8,
                            num_head=1,
                            attention_type='swin',
                            ffn_dim_expansion=4,
                            num_transformer_layers=6,
                            ).to(device)
            checkpoint = torch.load('PATH_TO/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth')
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            model.load_state_dict(weights)
            model.eval()
            inference_size = [480, 480] # Use larger inference size for better quality
            flow_horizon = 4
            shard = 4

        out = dict()

        out["images"] = process_images(path)
        # out["depth"] = process_depth(path)
        out["state"] = process_state(path)
        out["actions"] = process_actions(path)
        out["lang"] = process_lang(path)

        # data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
        date_time = datetime.strptime(path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")
        latency_shift = date_time < datetime(2021, 7, 23)

        # shift the actions according to camera latency
        if latency_shift:
            out["images"] = {k: v[1:] for k, v in out["images"].items()}
            out["state"] = out["state"][1:]
            out["actions"] = out["actions"][:-1]
            # if out["depth"] is not None:
            #     out["depth"] = out["depth"][1:]

        # append a null action to the end
        out["actions"].append(np.zeros_like(out["actions"][0]))

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]["images0"])

        # assemble episode
        episode = []
        episode_metadata = dict()

        # map original image name to correct image name according to logged camera topics
        orig_to_new = dict()
        for image_idx in range(len(out["images"])):
            orig_key = ORIG_NAMES[image_idx]

            if camera_topics[image_idx] in IMAGE_0_TOPICS:
                # fixed cam should always be image_0
                new_key = "image_0"
                # assert new_key[-1] == orig_key[-1], episode_path
            elif camera_topics[image_idx] == "/wrist/image_raw":
                # wrist cam should always be image_3
                new_key = "image_3"
            elif camera_topics[image_idx] in OTHER_TOPCIS:
                # other cams can be either image_1 or image_2
                new_key = "image_2" if "image_1" in list(orig_to_new.values()) else "image_1"
            else:
                raise ValueError(f"Unexpected camera topic {camera_topics[image_idx]}")

            orig_to_new[orig_key] = new_key
            episode_metadata[f"has_{new_key}"] = True

            if new_key == "image_0" and COMPUTE_FLOW:
                # compute flow for image_0
                image_0 = np.array(out["images"][orig_key])
                ori_size = image_0.shape[-3:-1]
                image_0 = torch.from_numpy(image_0).permute(0,3,1,2)
                image_0 = F.interpolate(image_0, size=inference_size, mode='bilinear', align_corners=True).to(device).float()
                image_0 = torch.cat((image_0, image_0[-1].unsqueeze(0).repeat(flow_horizon, 1, 1, 1)), dim=0)
                flows = []
                with torch.no_grad():
                    for i in range((image_0.shape[0] + shard - 1 - flow_horizon) // shard):
                        shard_start = i * shard
                        shard_end = min((i + 1) * shard, image_0.shape[0] - flow_horizon)
                        results_dict = model(image_0[shard_start:shard_end],
                                            image_0[shard_start + flow_horizon:shard_end + flow_horizon],
                                            attn_splits_list=[2],
                                            corr_radius_list=[-1],
                                            prop_radius_list=[-1],
                                            pred_bidir_flow=False,
                                            )
                        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
                        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
                        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                        flow_pr = flow_pr.permute(0, 2, 3, 1).cpu() # (B, H, W, 2)
                        flows.append(flow_pr)
                    flows = torch.cat(flows, dim=0).numpy()
                    out["image_0_flow"] = flows


        # record which images are missing
        missing_keys = set(NEW_NAMES) - set(orig_to_new.values())
        for missing in missing_keys:
            episode_metadata[f"has_{missing}"] = False

        instruction = out["lang"]

        for i in range(len(out["actions"])):
            observation = {
                "state": out["state"][i].astype(np.float32)
            }

            for orig_key in out["images"]:
                new_key = orig_to_new[orig_key]
                observation[new_key] = out["images"][orig_key][i]
            for missing in missing_keys:
                observation[missing] = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
            if COMPUTE_FLOW:
                if episode_metadata["has_image_0"]:
                    observation["image_0_flow"] = out["image_0_flow"][i]
                else:
                    observation["image_0_flow"] = np.zeros((*IMAGE_SIZE, 2), dtype=np.float32)

            episode.append(
                {
                    "observation": observation,
                    "action": out["actions"][i].astype(np.float32),
                    "is_first": i == 0,
                    "is_last": i == (len(out["actions"]) - 1),
                    "language_instruction": instruction,
                }
            )

        episode_metadata["file_path"] = path
        episode_metadata["has_language"] = bool(instruction)
        episode_metadata["ep_idx"] = ep_idx

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        # use episode path as key
        return path, sample

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # each path is a directory that contains dated directories
        paths = glob.glob(os.path.join(dl_manager.manual_dir, *("*" * (DEPTH - 1))))
        paths = [p for p in paths if "sink" in p]

        train_inputs, val_inputs = [], []

        for path in paths:
            for dated_folder in os.listdir(path):
                # a mystery left by the greats of the past
                if "lmdb" in dated_folder:
                    continue

                search_path = os.path.join(path, dated_folder, "raw", "traj_group*", "traj*")
                all_traj = glob.glob(search_path)
                if not all_traj:
                    print(f"no trajs found in {search_path}")
                    continue

                config_path = os.path.join(path, dated_folder, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                    camera_topics = config["agent"]["env"][1]["camera_topics"]
                else:
                    # assumed camera topics if no config.json exists
                    camera_topics = [
                        "/D435/color/image_raw",
                        "/blue/image_raw",
                        "/yellow/image_raw",
                        "/wrist/image_raw",
                    ]
                all_inputs = [(t, camera_topics) for t in all_traj]
                # shuffle the inputs
                random.shuffle(all_inputs)

                train_inputs += all_inputs

                val_inputs = train_inputs[N_TRAIN:]
                train_inputs = train_inputs[:N_TRAIN]

        logging.info(
            "Converting %d training and %d validation files.",
            len(train_inputs),
            len(val_inputs),
        )
        print(len(train_inputs), len(val_inputs))
        return {
            "train": self._generate_examples(train_inputs),
            "val": self._generate_examples(val_inputs),
        }

    def _generate_examples(self, inputs):
        """Yields examples."""
        for ep_idx, x in enumerate(inputs):
            yield Bridge._process_example(ep_idx, x)
