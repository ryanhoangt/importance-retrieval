"""
A script for importance sampling retrieval
"""

import collections
import math
import os
from pathlib import Path
import pickle
from pathlib import Path

import jax
import kde
import numpy as np
import scipy
import tensorflow as tf
import tqdm
from absl import app, flags
from jax import numpy as jnp
from jax.experimental import compilation_cache, multihost_utils
from matplotlib import pyplot as plt

from openx.data.dataloader import make_dataloader
from openx.utils.evaluate import load_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "The path to save the results if desired.", required=False)
flags.DEFINE_string("ckpt", None, "Path to the obs logs and checkpoints.", required=False)
flags.DEFINE_integer(
    "batch_size", 4096, "The batch size for the dataset, by default override the config.", required=False
)
flags.DEFINE_integer("repeat", 1, "The number of dataset repeats", required=False)
flags.DEFINE_enum(
    "reduction",
    "none",
    ["mean", "max", "min", "none"],
    "The way to reduce scores across multiple iterations",
    required=False,
)
flags.DEFINE_string("expert_dataset", None, "The name or path to the expert dataset.")
flags.DEFINE_string(
    "play_dataset", None, "The name of the play dataset in the original config, or None if there is only one."
)
flags.DEFINE_enum("smoothing", "none", ["none", "kernel3"], "Additional smoothing to scores within an episode")
flags.DEFINE_float(
    "retrieval_percentile", 70, "The percentile to use for retrieval statistics (100 - retrieval_percent)"
)
flags.DEFINE_bool("use_action", False, "Whether to use the first action in the retrieval.")
flags.DEFINE_bool("use_proprio", False, "Whether to use the last proprio in the retrieval.")
flags.DEFINE_float("bandwidth", 8, "The bandwidth to use for KDE estimation.")
flags.DEFINE_bool("only_expert_bw", False, "Whether or not to only modify the expert bw.")
flags.DEFINE_bool("use_inverse", False, "Whether or not to use explicit inverse calculation for the KDE.")

METADATA_KEYS = ["ep_idx", "step_idx", "quality_score", "task_id"]


class ScoreArray(object):
    """
    A dynamically resizing 2D array for efficiently storing scores.
    """

    def __init__(self, init_max_ep: int = 1000, init_max_steps: int = 400, reduction: str = "max"):
        assert reduction in {"mean", "min", "max", "none"}
        self.array = np.full((init_max_ep, init_max_steps), -np.inf, dtype=np.float32)
        self.reduction = reduction
        self._final = False

    def update(self, ep_idx, step_idx, v):
        assert not self._final, "Cannot update after getting"
        # First, see if we need to resize the array
        max_ep_idx, max_step_idx = np.max(ep_idx), np.max(step_idx)
        if max_ep_idx >= self.array.shape[0] or max_step_idx >= self.array.shape[1]:
            pad_ep = max(int(1.5 * self.array.shape[0]), max_ep_idx - self.array.shape[0] + 1)
            pad_step = max(int(1.5 * self.array.shape[1]), max_step_idx - self.array.shape[1] + 1)
            self.array = np.pad(self.array, ((0, pad_ep), (0, pad_step)), mode="constant", constant_values=-np.inf)
        # Write the values to the array
        selected = self.array[ep_idx, step_idx]
        if self.reduction == "none":
            self.array[ep_idx, step_idx] = v  # Simply set the value
        elif self.reduction == "max":
            self.array[ep_idx, step_idx] = np.maximum(selected, v)
        elif self.reduction == "min":
            self.array[ep_idx, step_idx] = np.where(selected != -np.inf, np.minimum(selected, v), v)
        elif self.reduction == "mean":
            self.array[ep_idx, step_idx] = np.where(selected != -np.inf, selected + v, v)
        else:
            raise ValueError("Invalid reduction type specified.")

    def get(self):
        self._final = True
        mask = self.array != -np.inf
        ep_lengths = np.sum(mask, axis=-1)
        last_idx = np.flatnonzero(ep_lengths)[-1]
        return self.array[: last_idx + 1, : np.max(ep_lengths) + 1]

def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # Define Shardings
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # Load Checkpoints
    alg, state, dataset_statistics, config = load_checkpoint(FLAGS.ckpt, sharding=rep_sharding)
    rng = jax.random.key(config.seed)

    # Define sharding and pad function.
    batch_size = FLAGS.batch_size

    def pad_batch(batch):
        valid = jax.tree.leaves(batch)[0].shape[0]
        if batch_size - valid > 0:
            num_repeats = math.ceil((batch_size - valid) / valid) + 1
            batch = jax.tree.map(
                lambda *xs: jnp.concatenate(xs, axis=0)[:batch_size], *[batch for _ in range(num_repeats)]
            )
        mask = np.zeros(batch_size, dtype=bool)
        mask[:valid] = True
        return batch, mask

    def unpad_batch_replicated(batch, mask):
        return jax.tree.map(lambda x: x[mask], batch)

    def shard_and_pad(batch):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        batch, valid = pad_batch(batch)
        return multihost_utils.host_local_array_to_global_array((batch, valid), mesh, dp_spec)

    # Create the structure and add indexing keys
    structure = config.structure.to_dict()
    for k in METADATA_KEYS:
        structure[k] = None

    # First determine the play dataset
    if FLAGS.play_dataset is None:
        if len(dataset_statistics) > 1:
            raise ValueError(
                "Trained on multiple datasets, must provide a dataset, choose one of: "
                + ", ".join(dataset_statistics.keys())
            )
        play_dataset = next(iter(dataset_statistics.keys()))
    else:
        play_dataset = FLAGS.play_dataset

    ### Phase 1: Embedding the expert demonstrations ###

    # Setup the expert dataset.
    expert_dataloader_config = config.dataloader.to_dict()
    expert_dataloader_config["repeat"] = False
    expert_dataloader_config["recompute_statistics"] = False
    expert_dataloader_config["global_dataset_statistics"] = None
    expert_dataloader_config["batch_size"] = FLAGS.batch_size
    if "augment_kwargs" in expert_dataloader_config:
        expert_dataloader_config["augment_kwargs"]["train"] = False
    expert_dataloader_config["drop_remainder"] = False
    if FLAGS.expert_dataset in expert_dataloader_config["datasets"]:
        # Remove all other datasets except the expert dataset if it was in the VAE config
        expert_dataloader_config["datasets"] = dict(expert=expert_dataloader_config["datasets"][FLAGS.expert_dataset])
    else:
        # Remove all datasets except the play dataset, and set it to a new config with the expert dataset
        # WARNING: this does not support using filters... it assumes the expert dataset is the whole chunk.
        play_ds_config = expert_dataloader_config["datasets"][play_dataset].copy()
        expert_ds_config = dict(
            path=FLAGS.expert_dataset,
            train_split="train",
            val_split=None,
            transform=play_ds_config["transform"],
            dataset_statistics=dataset_statistics[play_dataset],
            recompute_statistics=False,
        )
        expert_dataloader_config["datasets"] = dict(expert=expert_ds_config)
    expert_ds, _, _, _ = make_dataloader(**expert_dataloader_config, structure=structure, split_for_jax=False)
    expert_ds = map(shard_and_pad, expert_ds)

    # Define the embedding function and its jitted version
    def _embed(batch, rng):
        z = alg.predict(state, batch, rng)
        if FLAGS.use_action:
            # Use the first action
            z = jnp.concatenate((z, jnp.reshape(batch["action"][:, 0, :], (z.shape[0], -1))), axis=-1)
        if FLAGS.use_proprio:
            # Use the last joint state
            z = jnp.concatenate((z, jnp.reshape(batch["observation"]["state"][:, -1, :], (z.shape[0], -1))), axis=-1)
        return z

    jitted_embed_fn = jax.jit(_embed, in_shardings=(dp_sharding, None), out_shardings=rep_sharding)

    # Compute the expert zs
    expert_zs = []
    for i, (batch, mask) in tqdm.tqdm(enumerate(expert_ds), dynamic_ncols=True):
        rng = jax.random.fold_in(rng, i)
        z = jitted_embed_fn(batch, rng)
        z = unpad_batch_replicated(z, mask)  # TODO: this will fail on multi-host envs.
        expert_zs.append(z)

    expert_zs = jnp.concatenate(expert_zs, axis=0)

    # Determine Scott's factor and multiply by 8
    expert_bw_factor = FLAGS.bandwidth * expert_zs.shape[0] ** (-1 / (expert_zs.shape[1] + 4))
    kde_cls = jax.scipy.stats.gaussian_kde if FLAGS.use_inverse else kde.gaussian_kde
    expert_kde = kde_cls(expert_zs.T, bw_method=expert_bw_factor)

    ### Phase 2: Scoring ###

    # define the play dataset loader
    play_dataloader_config = config.dataloader.to_dict()
    play_dataloader_config["repeat"] = FLAGS.repeat if FLAGS.repeat > 1 else False
    play_dataloader_config["recompute_statistics"] = False
    play_dataloader_config["global_dataset_statistics"] = None
    play_dataloader_config["batch_size"] = FLAGS.batch_size
    if "augment_kwargs" in play_dataloader_config:
        play_dataloader_config["augment_kwargs"]["train"] = False
    play_dataloader_config["drop_remainder"] = False

    # Remove all datasets except the play dataset
    play_dataloader_config["datasets"][play_dataset]["dataset_statistics"] = dataset_statistics[play_dataset]
    play_dataloader_config["datasets"] = dict(play=play_dataloader_config["datasets"][play_dataset])
    # play_dataloader_config['datasets']['play']['path'] = '/iliad2/group/datasets/bridge_sink_rlds/bridge/1.0.0/'
    play_ds, _, _, _ = make_dataloader(**play_dataloader_config, structure=structure, split_for_jax=False)
    play_ds = map(shard_and_pad, play_ds)

    # For now we can write this function this way, but later might need to batch the score matrix
    # if (B_play, expert) is tooooo big.
    def _score(batch, mask, rng):
        play_z = alg.predict(state, batch, rng)
        if FLAGS.use_action:
            # Use the first action
            play_z = jnp.concatenate((play_z, jnp.reshape(batch["action"][:, 0, :], (play_z.shape[0], -1))), axis=-1)
        if FLAGS.use_proprio:
            # Use the last joint state
            play_z = jnp.concatenate(
                (play_z, jnp.reshape(batch["observation"]["state"][:, -1, :], (play_z.shape[0], -1))), axis=-1
            )

        # Compute the difference then the L2 Norm
        weights = mask.astype(jnp.float32) + 1e-7  # Add a small amount for stability
        neff = 1 / jnp.sum(weights**2)
        play_bw_factor = (1 if FLAGS.only_expert_bw else FLAGS.bandwidth) * neff ** (-1 / (play_z.shape[1] + 4))
        play_kde = kde_cls(play_z.T, bw_method=play_bw_factor, weights=weights)
        play_pdf = play_kde.logpdf(play_z.T)
        expert_pdf = expert_kde.logpdf(play_z.T)
        is_ratio = jnp.exp(expert_pdf - play_pdf)
        return is_ratio, {k: batch[k] for k in METADATA_KEYS if k in batch}, mask

    jitted_score_fn = jax.jit(
        _score, in_shardings=(dp_sharding, dp_sharding, None), out_shardings=(rep_sharding, rep_sharding, rep_sharding)
    )

    # Setup the scoring array, which lets us do all of the scoring in a faster numpy array.
    init_max_ep = int(dataset_statistics[play_dataset]["num_ep"].item())
    init_max_steps = 2 * int(dataset_statistics[play_dataset]["num_steps"].item()) // init_max_ep  # 2x Avg length
    score_array = ScoreArray(init_max_ep=init_max_ep, init_max_steps=init_max_steps, reduction=FLAGS.reduction)
    metadata_dict = collections.defaultdict(dict)

    for i, (batch, mask) in tqdm.tqdm(enumerate(play_ds), dynamic_ncols=True):
        rng = jax.random.fold_in(rng, i)
        scores, metadata, mask_replicated = jitted_score_fn(batch, mask, rng)
        scores, metadata = unpad_batch_replicated((scores, metadata), mask_replicated)

        # Log the scores -- note we explicitly fetch from device here
        ep_idxs, step_idxs = np.array(metadata["ep_idx"]), np.array(metadata["step_idx"])
        score_array.update(ep_idxs, step_idxs, np.array(scores))

        # Log the remaining metadata keys (if present). This will only be used for plotting.
        for k in metadata:
            if k != "step_idx" and k != "ep_idx":
                for v, ep_idx, step_idx in zip(np.array(metadata[k]), ep_idxs, step_idxs, strict=True):
                    metadata_dict[k][ep_idx, step_idx] = v

    score_array = score_array.get()  # (Num Ep, Num Steps). Reassign to allow garbage collect.
    if FLAGS.reduction == "mean" and FLAGS.repeat > 1:
        score_array /= FLAGS.repeat  # Divide by the repeat count to average.
    assert np.sum(np.isnan(score_array)) == 0, "Aborting, encountered NaN."
    mask = score_array == -np.inf

    ### Phase 3: Post-Processing ###

    # For IS we will apply some cliping to the most extreme ratios first.
    # This value cannot be higher since we will test retrieval at 0.5-1%
    score_array = np.clip(score_array, a_min=None, a_max=np.percentile(score_array[~mask].ravel(), 99.9))

    if FLAGS.smoothing != "none":
        kernel = {"kernel3": np.array([0.25, 0.5, 0.25])}[FLAGS.smoothing]
        min_vals = np.min(np.where(mask, np.inf, score_array), axis=-1)
        score_array = np.where(mask, min_vals[:, None], score_array)
        score_array = scipy.ndimage.convolve1d(score_array, kernel, axis=-1, mode="nearest")
        score_array[mask] = -np.inf  # Reset back

    # Now hash the array to a dictionary for saving.
    score_dict = dict()
    for idx, score in np.ndenumerate(score_array):
        if score != -np.inf:
            score_dict[idx] = score

    if os.path.dirname(FLAGS.path) != "":
        os.makedirs(os.path.dirname(FLAGS.path), exist_ok=True)
    with tf.io.gfile.GFile(FLAGS.path, "wb") as f:
        pickle.dump(score_dict, f)

    ### Phase 4: Visualization ###

    # Get the idxs of valid arrays where there is at least one step, then get associated ep_idxs
    ep_idxs = np.flatnonzero(np.sum(score_array == -np.inf, axis=-1) < score_array.shape[-1])
    score_array = score_array[ep_idxs]
    mask = score_array == -np.inf
    score_array[mask] = 0
    vae_name = Path(FLAGS.ckpt).parent.parent.name

    # First, plot a histogram of different score slices
    all_scores = score_array[~mask].ravel().copy()  # Explicitly copy for later use.
    plt.hist(all_scores)
    for p in [60, 80, 90, 95]:
        plt.axvline(np.percentile(all_scores, p), linestyle="--", color="red", label=f"{p}th %ile")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("All Scores histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"viz/{vae_name}_is_all_scores.png")
    plt.clf()

    ep_scores = np.sum(score_array, axis=1) / np.sum(~mask, axis=1)
    plt.hist(ep_scores)
    plt.xlabel("Ep Score")
    plt.ylabel("Count")
    plt.title("Ep Scores Histogram")
    plt.tight_layout()
    plt.savefig(f"viz/{vae_name}_is_ep_scores.png")
    plt.clf()

    step_scores = np.sum(score_array, axis=0) / np.sum(~mask, axis=0)
    plt.hist(step_scores)
    plt.xlabel("Step Score")
    plt.ylabel("Count")
    plt.title("Step Scores Histogram")
    plt.tight_layout()
    plt.savefig(f"viz/{vae_name}_is_step_scores.png")
    plt.clf()

    # Compute threshold and retrieved for later visualizations
    threshold = np.percentile(all_scores, FLAGS.retrieval_percentile)
    retrieved = (score_array >= threshold) & (~mask)

    # If we have the quality labels, do more advanced visualizations.
    if "quality_score" in metadata_dict:
        ep_qualities = np.array([metadata_dict["quality_score"][ep_idx, 0] for ep_idx in ep_idxs])
        sort_idx = np.argsort(ep_scores)  # Sorts from low score to high score episodes.

        rev_sorted_quality_labels = ep_qualities[sort_idx][::-1]  # The sorted quality labels
        total_quality_labels = np.cumsum(rev_sorted_quality_labels)
        num_data_points = 1 + np.arange(total_quality_labels.shape[0])
        avg_quality_label = total_quality_labels / num_data_points
        # Finally, re-reverse to set the axes back to num data points removed.
        plt.plot(np.arange(avg_quality_label.shape[0]), avg_quality_label[::-1], label="method")
        # Plot the oracle strategy
        oracle_labls = np.cumsum(np.sort(ep_qualities)[::-1]) / num_data_points
        plt.plot(np.arange(oracle_labls.shape[0]), oracle_labls[::-1], color="gray", label="oracle")
        plt.gca().hlines(np.mean(ep_qualities), xmin=0, xmax=oracle_labls.shape[0], color="red", linestyles="dashed")

        plt.xlabel("Episodes Removed")
        plt.ylabel("Average Quality Label")
        plt.legend(frameon=False)
        plt.ylim(np.min(ep_qualities), np.max(ep_qualities))
        plt.title("IS with KDE")
        plt.tight_layout()
        plt.savefig(f"viz/{vae_name}_is_ep_quality_curve.png")
        plt.clf()

        for quality_level in np.unique(ep_qualities):
            step_scores = np.sum(score_array[ep_qualities == quality_level], axis=0) / np.sum(
                ~mask[ep_qualities == quality_level], axis=0
            )
            plt.plot(step_scores, label=str(quality_level))

        plt.xlabel("Episode Step Index")
        plt.ylabel("Average Score")
        plt.legend(frameon=False)
        plt.title("IS with KDE")
        plt.tight_layout()
        plt.savefig(f"viz/{vae_name}_is_step_quality_curve.png")
        plt.clf()

        # We can also plot the retrieval amounts if desired.
        num_buckets = 10
        bucket_size = score_array.shape[-1] // num_buckets
        max_step_idx = bucket_size * num_buckets

        counts = []
        total = np.zeros(num_buckets)
        for quality_level in np.unique(ep_qualities):
            # get all eps of the given quality level
            count = retrieved[ep_qualities == quality_level][:, :max_step_idx]
            count = np.reshape(count, (count.shape[0], num_buckets, bucket_size))
            count = np.sum(count, axis=(0, 2))
            total += np.sum(count)  # Can change to just count for normalized per-bin
            counts.append(count)

        bottom = None
        for quality_level, count in zip(np.unique(ep_qualities), counts, strict=False):
            ratios = np.divide(count, total, where=total > 0)
            x = np.arange(num_buckets)
            bars = plt.bar(x, ratios, bottom=bottom, label=str(quality_level))
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{ratios[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )
            bottom = ratios

        plt.xlabel("Step Index Buckets")
        plt.ylabel("Ratio")
        plt.legend()
        plt.xticks(x, [f"{i * bucket_size}-{(i + 1) * bucket_size}" for i in range(num_buckets)], rotation=45)
        plt.savefig(f"viz/{vae_name}_is_retrieval_stats.png")
        plt.clf()

    if "task_id" in metadata_dict:  # Split retrieved data by task id and show differing percentages
        num_buckets = 10
        bucket_size = score_array.shape[-1] // num_buckets
        max_step_idx = bucket_size * num_buckets

        ep_task_ids = np.array([metadata_dict["task_id"][ep_idx, 0] for ep_idx in ep_idxs])
        task_counts = dict()
        step_counts = []
        for ep_task_id in np.unique(ep_task_ids):
            count = retrieved[ep_task_ids == ep_task_id]
            task_counts[ep_task_id.item()] = np.sum(count)
            count = np.reshape(count[:, :max_step_idx], (count.shape[0], num_buckets, bucket_size))
            count = np.sum(count, axis=(0, 2))
            step_counts.append(count)

        total = sum(task_counts.values())

        plt.bar(list(task_counts.keys()), list(task_counts.values()))
        plt.title("IS with KDE")
        plt.xlabel("Task ID")
        plt.ylabel("Num Retrieved")
        plt.savefig(f"viz/{vae_name}_is_retrieved_tasks_distribution.png")
        plt.clf()

        bottom = 0
        for ep_task_id, count in zip(np.unique(ep_task_ids), step_counts, strict=False):
            ratios = np.divide(count, total, where=total > 0)
            x = np.arange(num_buckets)
            bars = plt.bar(x, ratios, bottom=bottom, label=str(ep_task_id))
            bottom += ratios

        plt.title("IS with KDE")
        plt.xlabel("Step Index Buckets")
        plt.xticks(x, [f"{i * bucket_size}-{(i + 1) * bucket_size}" for i in range(num_buckets)], rotation=45)
        plt.ylabel("Num Retrieved")
        plt.legend()
        plt.savefig(f"viz/{vae_name}_is_task_by_step.png")
        plt.clf()

        num_tasks = 10
        # Subset to some number of tasks
        top_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:num_tasks]
        top_tasks = {k: v for k, v in top_tasks}

        plt.bar(list(top_tasks.keys()), list(top_tasks.values()))
        plt.title("IS with KDE")
        plt.xlabel("Task ID")
        plt.ylabel("Num Retrieved")
        plt.savefig(f"viz/{vae_name}_is_retrieved_tasks_distribution_top_{num_tasks}.png")
        plt.clf()

        top_total = sum(top_tasks.values())
        bottom = 0
        for ep_task_id, count in zip(np.unique(ep_task_ids), step_counts, strict=False):
            if ep_task_id.item() not in top_tasks:
                continue
            ratios = np.divide(count, top_total, where=top_total > 0)
            x = np.arange(num_buckets)
            bars = plt.bar(x, ratios, bottom=bottom, label=str(ep_task_id))
            bottom += ratios

        plt.title("IS with KDE")
        plt.xlabel("Step Index Buckets")
        plt.xticks(x, [f"{i * bucket_size}-{(i + 1) * bucket_size}" for i in range(num_buckets)], rotation=45)
        plt.ylabel("Num Retrieved")
        plt.legend()
        plt.savefig(f"viz/{vae_name}_is_task_by_step_top_{num_tasks}.png")
        plt.clf()

if __name__ == "__main__":
    app.run(main)
