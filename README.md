# Neural Loop Algorithm

Reinforcement learning framework for discovering Monte Carlo algorithm on topological model.

![](https://youtu.be/8ElDneQO1ac)

## Prerequisites
General
* CMake (>= 2.8.3)
* Boost (>= 1.3.2)
* Python 2.7
* GCC

Python
* matlotlib
* Tesnorflow 1.4

For Mac OSX, we need to install extra boost-python library.
```
brew install cmake boost-python
```
for more details, please refer to https://github.com/TNG/boost-python-examples

## Installation

1. Compile icegame core
2. Install gym-icegame interface (follow instructions in icegame2)


## Inference
The inference should be executed at the folder rlloop. Go to the folder and download the trained model.
```
cd a3c
sh download.sh
```

or download model from https://drive.google.com/drive/folders/15MO-S_po4NIKsBL94rhOG5rC-fMbBn18?usp=sharing.

Now, we can play with it.
```
python2.7 play_icegame.py --log-dir saved_model
```
Use `--render` for visualization.

## Training

The following command will launch 8 workers 1 parameter server and 1 rewards monitor.
```
python distribute_tasks.py -w 8 -l logs/my_task
```

## Experiment Settings
For training, it takes about 3 days on 12 cpu cores.


## Physical Observables Measurement
The code in a3c_measure/ folder is modified to measure the correlation function, structure factor, probability frequency.

Download the Models and put in the folder to run. 

```
    sh run_*.sh
```



