import functools
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

COMPUTE_FLOW = False
if COMPUTE_FLOW:
    import torch
    from gmflow.gmflow.gmflow import GMFlow
    import torch.nn.functional as F

def _chain_with_global_idx(*gens):
    num = 0
    for gen in gens:
        for x in gen(idx=num):
            yield x
            num += 1


def _is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.

    Taken from https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py

    Note: this is currently not used because it seemed to hurt performance.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


class Libero(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "You can download the raw robomimic datasets from https://github.com/Lifelong-Robot-Learning/LIBERO."
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
                                    "agent_image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "agent_flow": tfds.features.Tensor(
                                        shape=(128, 128, 2),
                                        dtype=np.float32,
                                        doc="Main camera flow",
                                    ),
                                    "wrist_flow": tfds.features.Tensor(
                                        shape=(128, 128, 2),
                                        dtype=np.float32,
                                        doc="Wrist cameraflow.",
                                    ),
                                    "state": tfds.features.FeaturesDict(
                                        {
                                            "ee_pos": tfds.features.Tensor(
                                                shape=(3,), dtype=np.float32, doc="Robot EEF Position"
                                            ),
                                            "ee_euler": tfds.features.Tensor(
                                                shape=(3,), dtype=np.float32, doc="Robot EEF EULER"
                                            ),
                                            "gripper_qpos": tfds.features.Tensor(
                                                shape=(2,), dtype=np.float32, doc="Robot EEF Quat"
                                            ),
                                            "joint_pos": tfds.features.Tensor(
                                                shape=(7,),
                                                dtype=np.float32,
                                                doc="Robot joint angles.",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot EEF action.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount if provided, default to 1."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32, doc="Reward if provided, 1 on final step for demos."
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "ep_idx": tfds.features.Scalar(
                                dtype=np.int32, doc="Detemrinistic index of the episode in the entire dataset."
                            ),
                            "demo_idx": tfds.features.Scalar(
                                dtype=np.int32, doc="Detemrinistic index of the demonstration in the file."
                            ),
                            "task": tfds.features.Text(doc="The name of the task."),
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
                                    "agent_image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.FeaturesDict(
                                        {
                                            "ee_pos": tfds.features.Tensor(
                                                shape=(3,), dtype=np.float32, doc="Robot EEF Position"
                                            ),
                                            "ee_euler": tfds.features.Tensor(
                                                shape=(3,), dtype=np.float32, doc="Robot EEF EULER"
                                            ),
                                            "gripper_qpos": tfds.features.Tensor(
                                                shape=(2,), dtype=np.float32, doc="Robot EEF Quat"
                                            ),
                                            "joint_pos": tfds.features.Tensor(
                                                shape=(7,),
                                                dtype=np.float32,
                                                doc="Robot joint angles.",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot EEF action.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount if provided, default to 1."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32, doc="Reward if provided, 1 on final step for demos."
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "ep_idx": tfds.features.Scalar(
                                dtype=np.int32, doc="Detemrinistic index of the episode in the entire dataset."
                            ),
                            "demo_idx": tfds.features.Scalar(
                                dtype=np.int32, doc="Detemrinistic index of the demonstration in the file."
                            ),
                            "task": tfds.features.Text(doc="The name of the task."),
                        }
                    ),
                }
            )
        return self.dataset_info_from_configs(
            features=features
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        Define filepaths for data splits.
        Modify this at each call.
        """
        dataset_files = [
            tf.io.gfile.join(dl_manager.manual_dir, f)
            for f in tf.io.gfile.listdir(dl_manager.manual_dir)
            if f.endswith(".hdf5")
        ]
        generators = [functools.partial(self._generate_examples, path=f) for f in dataset_files]

        return {"train": _chain_with_global_idx(*generators)}

    def _generate_examples(self, path: str, idx: int = 0) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
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
            flow_horizon = 8
            shard = 128

        print("Trying to open", path)
        f = h5py.File(path, "r")

        # Get the language, scene, and task from the filename.
        task = os.path.basename(path)[: -len("_demo.hdf5")]

        # Convert the task to language instruction
        language_instruction = ""
        for w in task.split("_"):
            if "SCENE" in w:
                language_instruction = ""
                continue
            language_instruction = language_instruction + w + " "
        language_instruction = language_instruction[:-1]

        # Start ep_idx count
        ep_idx = idx

        for demo_idx in range(len(f["data"])):
            demo = "demo_" + str(demo_idx)
            demo_length = f["data"][demo]["dones"].shape[0]
            # Remember to fix the images by fliping them on the H axis.
            data = dict(
                action=f["data"][demo]["actions"][:].astype(np.float32),
                observation=dict(
                    agent_image=np.flip(f["data"][demo]["obs"]["agentview_rgb"][:], axis=1),  # (B, H, W, C)
                    wrist_image=np.flip(f["data"][demo]["obs"]["eye_in_hand_rgb"][:], axis=1),
                    state=dict(
                        ee_pos=f["data"][demo]["obs"]["ee_pos"][:].astype(np.float32),
                        ee_euler=f["data"][demo]["obs"]["ee_ori"][:].astype(np.float32),
                        gripper_qpos=f["data"][demo]["obs"]["gripper_states"][:].astype(np.float32),
                        joint_pos=f["data"][demo]["obs"]["joint_states"][:].astype(np.float32),
                    ),
                ),
                is_first=np.zeros(demo_length, dtype=np.bool_),
                is_last=np.zeros(demo_length, dtype=np.bool_),
                is_terminal=np.zeros(demo_length, dtype=np.bool_),
                discount=np.ones(demo_length, dtype=np.float32),
                reward=f["data"][demo]["rewards"][:],
            )

            if COMPUTE_FLOW:
                image_flows = dict()
                for image_key in ['agent_image', 'wrist_image']:
                    rm_image = data['observation'][image_key] # (B, H, W, C)
                    ori_size = rm_image.shape[-3:-1]
                    rm_image = torch.from_numpy(rm_image.copy()).permute(0, 3, 1, 2)
                    rm_image = F.interpolate(rm_image, size=inference_size, mode='bilinear', align_corners=True).to(device).float()
                    rm_image = torch.cat((rm_image, rm_image[-1].unsqueeze(0).repeat(flow_horizon, 1, 1, 1)), dim=0)
                    flows = []
                    with torch.no_grad():
                        for i in range((rm_image.shape[0] + shard - 1 - flow_horizon) // shard):
                            shard_start = i * shard
                            shard_end = min((i + 1) * shard, rm_image.shape[0] - flow_horizon)
                            results_dict = model(rm_image[shard_start:shard_end],
                                                rm_image[shard_start + flow_horizon:shard_end + flow_horizon],
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
                    image_flows[image_key] = flows
                    data['observation'][image_key.replace('image', 'flow')] = image_flows[image_key]

            # Convert into steps.
            episode = []
            for i in range(demo_length):
                step = tf.nest.map_structure(lambda x, i=i: x[i], data)
                step["language_instruction"] = language_instruction
                episode.append(step)

            # Add is first label
            episode[0]["is_first"] = True

            # Finally add the terminal states.
            # Note that LIBERO does not save final states..... so we duplicate the last observation
            # This should be thrown away by the dataloader in most cases.
            terminal_step = dict(
                action=np.zeros(7, dtype=np.float32),
                observation=episode[-1]["observation"],
                is_first=False,
                is_last=True,
                is_terminal=True,
                discount=1.0,
                reward=1.0,  # Assume terminal rewards are 1
                language_instruction=language_instruction,
            )
            episode.append(terminal_step)

            metadata = dict(ep_idx=ep_idx, demo_idx=demo_idx, task=task, file_path=os.path.basename(path))
            yield ep_idx, dict(steps=episode, episode_metadata=metadata)

            # Increment the ep idx
            ep_idx += 1

        # Finally close the file.
        f.close()
