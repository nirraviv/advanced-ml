## Connecting to the machine

If you are outside of the University, you will need to first connect to NOVA:

```BASH
ssh USERNAME@nova.cs.tau.ac.il
```

You can then connect to the savant machine:

```BASH
ssh savant
```

All of our files are located in `/home/gamir/leshchenko`

## Using tmux

For executing long running tasks it is useful to use the tmux terminal multiplexer.
You can create remote sessions that continue running after you disconnect,
and which you can pick up later.

## Creating a Python virtual environment

Create the virtual environment using:

```BASH
python3.6 -m venv MY_ENV
```

You can then activate it using:

```BASH
source MY_ENV/bin/activate     # for bash
source MY_ENV/bin/activate.csh # for csh, the default shell on savant
```

Now, you can install packages using `pip install`, and use them by just running `python`. Many projects provide their dependencies in a requirements file, making them easy to install.

```BASH
pip install MY_PACKAGE
python MY_SCRIPT
pip install -r requirements.txt
```

Finally, you can leave the virtual environment using:

```BASH
deactivate
```

## Running a script on a GPU

Before the first run you will need to add the following to your ~/.tcshrc file (and source it):

```BASH
setenv LD_LIBRARY_PATH /usr/local/lib/cuda-9.0.176/lib64:/usr/local/lib/cuda_8.0.61/lib64/:/usr/local/lib/cudnn-8.0-v7/lib64/
```

Before each run, check which GPUs are used by running:

```BASH
nvidia-smi
```

Select one (remember that Amir asked us to not use GPU number 0), and then execute your script (in **tcsh**):

```BASH
setenv CUDA_VISIBLE_DEVICES GPU_NUMBER
python MY_SCRIPT
```

or in **bash**:
```BASH
export CUDA_VISIBLE_DEVICES=GPU_NUMBER
python MY_SCRIPT
## OR
CUDA_VISIBLE_DEVICES=GPU_NUMBER python MY_SCRIPT
```

You can monitor the GPU usage using `nvidia-smi`, but a better option is to install `gpustat` through `pip`.
Then you can interactively monitor the GPU usage using:

```BASH
watch --color -n1.0 gpustat --color
```

## Training tensorflow-wavenet

1. Clone https://github.com/ibab/tensorflow-wavenet using Git.
2. Create a virtual environment, and install the requirements file.
3. Run the code as shown in the documentation.
4. Use Tensorboard to observe the traning process.

Example commands for training and generating

```BASH
CUDA_VISIBLE_DEVICES=6 python train.py --data_dir=../../data/damp/sliced_us_01/ --silence_threshold=0.3

CUDA_VISIBLE_DEVICES=6 python generate.py logdir/train/2018-08-12T19-57-48/model.ckpt-18000 --samples 16000 --wav_out_path=results/result01.wav
```

## Training Tacotron-1

1. Clone https://github.com/keithito/tacotron using Git
2. Create a virtual environment, and install the requirements file.

```BASH
# Link the dataset as LJ speech
ln -s /home/gamir/leshchenko/data/damp/02_us_f_ljspeech LJSpeech-1.1

# Preprocess
python preprocess.py --base_dir $PWD --num_workers 8 --dataset ljspeech

# Begin training from scratch
CUDA_VISIBLE_DEVICES=7 python train.py --base_dir $PWD

# Fine-tune existing model
mkdir ready_models
curl http://data.keithito.com/data/speech/tacotron-20170720.tar.bz2 | tar xjC ready_models
... 'Change some stuff in the train.py file'
CUDA_VISIBLE_DEVICES=2 python train.py --base_dir $PWD --restore_path ready_models/tacotron-20170720/model.ckpt
CUDA_VISIBLE_DEVICES=2 python train.py --base_dir $PWD --restore_path ready_models/nancy/model.ckpt-281000

CUDA_VISIBLE_DEVICES=6 python train.py --base_dir $PWD --name=taco_slow_e1 --hparams=initial_learning_rate=0.0002 --restore_step=284000

CUDA_VISIBLE_DEVICES=4 python train.py --base_dir $PWD --restore_path ready_models/nancy250KGS/model.ckpt-250000
CUDA_VISIBLE_DEVICES=6 python train.py --base_dir $PWD --name=taco_slow_e1 --hparams=initial_learning_rate=0.0002 --restore_path ready_models/nancy250KGS/model.ckpt-250000
CUDA_VISIBLE_DEVICES=7 python train.py --base_dir $PWD --name=taco_slow_e2 --hparams=initial_learning_rate=0.00002 --restore_path ready_models/nancy250KGS/model.ckpt-250000

TODO(Andrey): Rewrite

# Generate audio
CUDA_VISIBLE_DEVICES=6 python demo_server.py --checkpoint logs-tacotron/model.ckpt-282000

```

## Training Tacotron-2

1. Clone https://github.com/Rayhane-mamah/Tacotron-2 using Git
2. Create a virtual environment and install the requirements file, OR link Nir''s environment

```BASH
# Link the dataset as LJ speech
ln -s /home/gamir/leshchenko/data/gregorian_chants/chants_03_ljspeech LJSpeech-1.1

# Preprocess
python preprocess.py --base_dir $PWD --n_jobs 8 --dataset LJSpeech-1.1

# Begin training from scratch
CUDA_VISIBLE_DEVICES=3 python train.py --base_dir $PWD --model='Tacotron-2'


CUDA_VISIBLE_DEVICES=4 python synthesize.py --model='Tacotron-2' --text_list=my_sentences.txt

```
