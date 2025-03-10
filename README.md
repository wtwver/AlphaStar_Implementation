python run_sl.py --workspace_path ~/AlphaStar_Implementation/ --model_name fullyconv --training True --gpu_use True --learning_rate 0.0001 --replay_hkl_file_path ~/AlphaStar_Implementation/hkl/h/ --environment Simple64

python trajectory_generator.py --replay_path ../replay/ --saving_path ../pkl/1/

# Install
uv venv --python=3.9
uv pip install tensorflow[and-cuda] matplotlib tf-keras pysc2==3.0 sci-klearn hickle

# Introduction
This repository is for Deep Learning agent of Starcraft2. It is very similar to AlphaStar of DeepMind except size of network. I only test my code with Minigame, Simple64 map of PySC2. However, I am sure this code will work at more large scale game if network size is grown.

# Reference
1. Download replay file(4.8.2 version file is needed): https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api
2. Extracting observation, action from replay file: https://github.com/narhen/pysc2-replay
3. FullyConv model of Tensorflow 1 version: https://github.com/simonmeister/pysc2-rl-agents
4. Supervised Learning technique: https://github.com/metataro/sc2_imitation_learning/tree/8dca03e9be92e2d8297a4bc34248939af5c7ec3b

# Version
## Python
1. Python3.7 or 3.8
2. PySC2 3.0.0: https://github.com/deepmind/pysc2
6. Pygame 1.9.6
7. Sklearn
8. ZeroMQ

## Starcraft2
1. Client 4.8.2: https://github.com/Blizzard/s2client-proto#downloads
2. Replay 4.8.2

## PC capaticy
1. NVIDIA RTX A6000 x 1
2. 128GB RAM
3. Ubuntu 20.04

# Comment for code
Instead of adding a comment to the code, an overall explanation about the code is written in Medium. Please visit using the link at the bottom of the page.

# Network architecture
## FullyConv
<img src="image/network_architecture(fullyconv).png" width="1000">

## AlphaStar
<img src="image/network_architecture.png" width="1000">

# Notice
There may be a minor error such as a GPU setting, unit list, and network size. However, you can run it without major modification because I checked that the latest code works for Supervised, Reinforcement Learning. It is not easy to check every part of the code because it is huge.

# Reinforcement Learning
I can only check that the FullyConv model works well in Reinforcement Learning. The LSTM model takes too much time for training and does not show better performance than FullyConv yet.

In the case of RL, the training speed is improved by introducing [IMPALA](https://arxiv.org/abs/1802.01561) of DeepMind, which separates the learner and actor.
<img src="image/impala_architecture.png" width="400">

To run that training method, you first run the learner file using the below command.
```
$ python learner.py --env_num 4 --gpu_use True --model_name fullyconv  --gradient_clipping 10.0
```

You can ignore the below error of the learner.py part. It does not affect the training process. Please check the actor.py is running well.

```
Traceback (most recent call last):
File "C:/minerl/learner.py", line 392, in
coord.join(thread_data)
File "C:\Users\sund0\anaconda3\envs\minerl_env\lib\site-packages\tensorflow\python\training\coordinator.py", line 357, in join
threads = self._registered_threads.union(set(threads))

where line 391 and 392 is
for thread_data in thread_data_list:
coord.join(thread_data)
```

Next, you should run the multiple actor based on the number of env_num of the Learner. They should be run from separate terminals and can be distinguished from env_id.
```
$ python actor.py --env_id 0 --environment CollectMineralShards
$ python actor.py --env_id 1 --environment CollectMineralShards
$ python actor.py --env_id 2 --environment CollectMineralShards
$ python actor.py --env_id 3 --environment CollectMineralShards
```

I also provide the bash file to run the below process using [tmux](https://github.com/tmux/tmux/wiki). You can start the leaner and actors using a single terminal.
```
$ ./run_reinforcement_learning.sh 8 True CollectMineralShards fullyconv
```

You can also terminate the learner and actors using the bash script.
```
$ ./stop.sh
```

## Gradient Clipping
Gradient clipping is essential for training the model of PySC2 because it has multiple stae encoder and action head network. In my experience, the gradient norm value is changed based on network size. Therefore, you should check it every time you change the model structure. You can check it by using 'tf.linalg.global_norm' function.

```
grads = tape.gradient(loss, model.trainable_variables)
grad_norm = tf.linalg.global_norm(grads)
tf.print("grad_norm: ", grad_norm)
grads, _ = tf.clip_by_global_norm(grads, arguments.gradient_clipping)
```

<img src="image/gradient_clipping.png" width="400">

After checking the norm value, you should remove an outlier value among them.

## Stacked Screen Observation
One of the big differences between turn-based games like Go and Real-time strategy games is that there is no single state to determine whether a unit is approaching or moving away from a specific point. Therefore, you should either use LSTM as in DeepMind's [Relational Deep Reinforcement Learning paper](https://arxiv.org/pdf/1806.01830.pdf), or keep the states of the previous step and use them as the current state.
_____________________________________________________________________
<img src="image/stacked_observation_deepmind.png" width="800">

In this project, the latter method was used, and it seemed to have the same effect as ConvLSTM3D.

<img src="image/stacked_observation_mine.png" width="200">

## CollectMineralShards
First, let's test the sample code for the MoveToBeacon environment which is the simplest environment in PySC2 using a model which has a similar network structure as AlphaStar. First, run 'git clone https://github.com/kimbring2/AlphaStar_Implementation.git' command in your workspace. Next, start training by using the below command. 

```
$ ./run_reinforcement_learning.sh 1 True CollectMineralShards fullyconv
```

I provide a FullyConv, AlphaStar style model. You can change a model by using the model_name argument. The default is FullyConv model.

After the training is completed, test it using the following command. Training performance is based on two parameters. Try to use 100.0 as the gradient_clipping and 0.0001 as the learning_rate. Furthermore, training progress and results depend on the seed value. The model is automatically saved if the average reward is improved.

<img src="image/CollectMineralShards_IMPALA.png" width="400">

After finishing training, run the below command to test the pre-trained model that was saved under the Models folder of the workspace. 

```
$ python run_evaluation.py --workspace_path /home/kimbring2/pysc2_impala --visualize True --gpu_use True --pretrained_model reinforcement_model_13626481 --environment CollectMineralShards
```

If the accumulated reward is over 100 per episode, you can see the Marines collect the Mineral well.

<img src="image/alphastar_mineral.gif" width="800">

# Supervised Learning 
I can only check that the model with LSTM works well in Supervised Learning. The FullyConv model does not show good performance yet although it fasts the LSTM model for training. 

## Simple64
To implement AlphaStar successfully, Supervised Training is crucial. Instead of using the existing replay data to check a simple network of mine, I collect the amount of 1000 [replay files](https://drive.google.com/drive/folders/1Tdt-7LaQWQijT7MZWYCr5fGn1CECUsFa?usp=sharing) in Simple64 map using only Terran, and Marine rush from two Barrack with Random race opponent.

First, change a Starcraft2 replay file to hkl file format for fast training. It will remove a step of no_op action except when it occurred at first, end of the episode, and 8 dividble steps. You need around 80GB disk space to convert several around 1000 replay files to hkl. Currently, I only use replay files of Terran vs Terran.
```
$ python trajectory_generator.py --replay_path [your path]/StarCraftII/Replays/local_Simple64/ --saving_path [your path]/pysc2_dataset/simple64
```

After making hkl file of replay in your workspace, try to start the Supervised Learning using the below command. It will save a trained model under the Models folder of your workspace.

```
$ python run_supervised_learning.py --workspace_path [your path]/AlphaStar_Implementation/ --model_name alphastar --training True --gpu_use True --learning_rate 0.0001 --replay_hkl_file_path [your path]/pysc2_dataset/simple64/ --environment Simple64 --model_name alphastar
```

You can check training progress using Tensorboard under the Tensorboard folder of your workspace. It will take a very long time to finish training because of the vast observation and action space.

<img src="image/SL_Tensorboard.png" width="600">

Below is the code for evaluating the trained model

```
python run_evaluation.py --workspace_path [your path]/AlphaStar_Implementation/ --gpu_use True --visualize True --environment Simple64 --pretrained_model supervised_model --model_name alphastar
```

Video of Downisde is one behavior example of a trained agent.

[![Supervised Learning demo](https://img.youtube.com/vi/ABomHc4_GlQ/maxresdefault.jpg)](https://youtu.be/ABomHc4_GlQ "AlphaStar Implementation - Click to Watch!")
<strong>Click to Watch!</strong>

I only use a replay file of the Terran vs Terran case. Therefore, the agent only needs to recognize 19 units during the game. It can make the size of the model not need to become huge. The total unit number of Starcraft 2 is over 100 in full-game cases. For that, we need a more powerful GPU to run.

# Detailed information
I am writing an explanation for code at Medium as a series.

1. Tutorial about Replay file: https://medium.com/@dohyeongkim/alphastar-implementation-serie-part1-606572ddba99
2. Tutorial about Network: https://dohyeongkim.medium.com/alphastar-implementation-series-part5-fd275bea68b5
3. Tutorial about Reinforcement Learning: https://medium.com/nerd-for-tech/alphastar-implementation-series-part6-4044e7efb1ce
4. Tutorial about Supervised Learning: https://dohyeongkim.medium.com/alphastar-implementation-series-part7-d28468c07739
