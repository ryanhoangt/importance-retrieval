import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

from openx.data.filters import task_filter

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "The path to save the ep idx pickle file")
flags.DEFINE_string("dataset", None, "The path to the dataset")
flags.DEFINE_integer("num", None, "Number of datapoints in the filter.")
flags.DEFINE_string("task", None, "The name of the libero task.")

DEMO_IDXS = [4, 10, 12, 24, 39]  # Fixed for all tasks.


def main(_):
    builder = tfds.builder_from_directory(builder_dir=FLAGS.dataset)
    ds = builder.as_dataset(
        split="train",
        decoders=dict(steps=tfds.decode.SkipDecoding()),
        shuffle_files=True,
        read_config=tfds.ReadConfig(
            skip_prefetch=True,
            shuffle_reshuffle_each_iteration=True,
        ),
    )
    ds = ds.filter(task_filter(FLAGS.task))

    tf.io.gfile.makedirs(FLAGS.path)

    ds_identity = tfds.core.dataset_info.DatasetIdentity(
        name=builder.info.name + "_subset", version=tfds.core.Version("1.0.0"), data_dir=FLAGS.path, module_name=""
    )

    ds_info = tfds.core.DatasetInfo(
        builder=ds_identity,
        description=builder.info.description,
        features=builder.info.features,
        supervised_keys=builder.info.supervised_keys,
        homepage=builder.info.homepage,
        citation=builder.info.citation,
        metadata=builder.info.metadata,
    )

    sequential_writer = tfds.core.SequentialWriter(
        ds_info,
        10,
        overwrite=True,  # Overwrites existing data
        file_format=builder.info.file_format,
    )
    sequential_writer.initialize_splits(["train"], fail_if_exists=True)

    for ep in ds:
        ep_np = tf.nest.map_structure(lambda x: x._numpy(), ep)
        if ep["episode_metadata"]["demo_idx"] in DEMO_IDXS:
            sequential_writer.add_examples({"train": [ep_np]})

    sequential_writer.close_all()  # Close the writer.


if __name__ == "__main__":
    app.run(main)
