import pickle

import numpy as np
import tensorflow as tf


def filter_by_ep_path(search_strings, path_key: str = "file_path"):
    def _filter(ep):
        string = ep["episode_metadata"][path_key]
        if isinstance(search_strings, list):
            bool_tensor = tf.concat(
                [tf.strings.regex_full_match(string, pattern=".*" + s + ".*") for s in search_strings], axis=0
            )
            return tf.math.reduce_all(bool_tensor)
        return tf.strings.regex_full_match(string, pattern=".*" + search_strings + ".*")

    return _filter


def quality_filter(threshold):
    def _filter(ep):
        return tf.cast(ep["episode_metadata"]["quality_score"] >= threshold, tf.bool)

    return _filter


def task_filter(task):
    def _filter(ep):
        return tf.cast(ep["episode_metadata"]["task"] == task, tf.bool)

    return _filter


def retrieval_filter(path, percentile):
    with tf.io.gfile.GFile(path, "rb") as f:
        scores = pickle.load(f)

    threshold = np.percentile(np.array(list(scores.values())), percentile)
    keys = [k for k, v in scores.items() if v >= threshold]  # Only keep values above threshold
    max_step_idx = max(k[1] for k in scores) + 2  # Add an offset to make idxs unique.
    keys = [max_step_idx * ep_idx + step_idx for ep_idx, step_idx in keys]

    binary_set_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys, dtype=tf.int32),  # Cast to int32
            values=tf.constant(tf.ones(len(keys), dtype=tf.int32)),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    def _filter(step):
        key = step["ep_idx"] * max_step_idx + step["step_idx"]
        return tf.cast(binary_set_lookup.lookup(key), tf.bool)

    return _filter
