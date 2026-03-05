# Data Retrieval with Importance Weights for Few-Shot Imitation Learning

Official Implementation for "Data Retrieval with Importance Weights for Few-Shot Imitation Learning" by Amber Xie, Rahul Chand, Dorsa Sadigh, Joey Hejna. 

Code for this project is based on the [OpenX Repository](https://github.com/jhejna/openx), which is heavily based upon the [Octo repository](https://github.com/octo-models/octo).

## Installation
First, create a conda environment with python 3.11, and then install requirements and this repo.
```
conda create -n openx python=3.11
pip install -r requirements.txt
pip install -e .
```
If you are on GPU, you will additionally need to install the corresponding jaxlib verison.
```
pip install --upgrade "jax[cuda12_pip]==0.4.37" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If you are on TPU, instead run:
```
pip install --upgrade "jax[tpu]==0.4.37" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Note that you will need to use separate conda environments for robomimic and libero.

### Robomimic
We benchmarked some of our implementations against Pytorch versions in robomimic. Installing the correct robomimic version corresponding to that used in the [original Robomimic paper](https://arxiv.org/abs/2108.03298) is pain. We provide more details commented out in the requirements.txt file, but the basics are as follows.

First, follow the instructions to install `mujoco210_linux` found [here](https://github.com/openai/mujoco-py)

```
sudo apt-get update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf # If error use the command below
sudo apt-get install -y libgl1
```

Then, install robosuite, robomimic, and needed dependencies.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies
pip install "mujoco-py<2.2,>=2.0"
pip install cython==0.29.37
pip install numba

# Robosuite
git clone https://github.com/ARISE-Initiative/robosuite/
cd robosuite
git checkout offline_study
pip install -e . --no-deps # Ignore
cd ..

# Robomimic
git clone https://github.com/ARISE-Initiative/robomimic/
cd robomimic
git checkout v0.2.0
pip install -e . --no-deps # Ignore
cd ..
```
and enable `USE_MUJOCO_PY` in `setup_shell.sh`.

Then repeatedly try to import mujoco_py, robosuite, and robomimic until it works. There are a few manual changes to the code in robosuite and robomimic you will need to make:
1. Comment out all references to EGL Probe if you are using TPU.
2. You will need to change some imports to `from collections.abc` from `from collections`. This is because some typing hints used in robosuite and robomimic were deprecated in Python 3.11.

### Libero
If you want to use the libero benchmark, you have to follow separate installation instructions. Note that we parse these dependencies out carefully to prevent conflicts. For example, we make sure to install the CPU only version of PyTorch.

For TPUs, ensure the following are installed:
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libsm6 libxext6
```

Then install the following python dependencies (in this order):
```
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install robosuite==1.4.0 bddl==1.0.1 future "easydict==1.9" termcolor

git clone https://github.com/Lifelong-Robot-Learning/LIBERO
cd LIBERO
pip install -e . --no-deps
```

To avoid installing gym, I then comment out the line `from .venv import SubprocVectorEnv, DummyVectorEnv` in `LIBERO/libero/libero/envs/venv.py`.

If you encounter an error relating to `AttributeError: 'NoneType' object has no attribute 'glGetError'` when using `MUJOCO_GL="osmesa"` try the following fix:

When creating your base conda environment, use the following command.
```
conda install -c conda-forge libstdcxx-ng
```
and do not enable `USE_MUJOCO_PY` in `setup_shell.sh`.

## Usage

You can train a model with
```
python scripts/train.py --config path/to/config:config_str --path save/path --name name/on/wandb --project project/on/wandb
```

Example config files can be found in `configs`.

Sample commands to train a VAE, compute retrieval statistics, and co-train a BC agent. Make sure to set dataset, model, and retrieval paths in the config files, which we have left as `PATH/TO`

**Step 1:** Train the VAE
```
python scripts/train.py --config configs/vae/robomimic_vae_image.py --path save/path
```

**Step 2:** Run retrieval with is.py or br.py (note: SAILOR, FlowRet use br.py too)
```
python scripts/retrieval/is.py --path path/to/save/retrieval/stats.pkl --ckpt path/to/vae/ckpt --expert_dataset path/to/rlds/expert/dataset --dataset square_400_paired
```

**Step 3:** Train a policy using the retrieval file
```
python scripts/train.py --config configs/bc/robomimic_image_retrieval.py:<path/to/retrieval/>,<retrieval%>,<seed> --path save/path
```

### Generating Datasets
To generate RLDS datasets, follow the instructions in `rlds`. 

* For Libero, we download the libero datasets and convert using our provided code. 
* For RoboMimic, we use the [code from Behavior Retrieval](https://github.com/MaxDu17/BehaviorRetrieval) to generate the 50-50 square dataset.
* For bridge, we download the [raw bridge v2 dataset](https://rail-berkeley.github.io/bridgedata/) and convert the kitchen scenes.

To train FlowRetrieval, we use [FlowRetrieval's code](https://github.com/lihenglin/bridge_training_code/tree/main) for GMFlow and generate an RLDS dataset. Make sure to set the flag `COMPUTE_FLOW=True` if generating flow, and False if not.
