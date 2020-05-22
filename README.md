# trajectory_prediction_TraPHic
This repository for studying [TraPHic paper (CVPR '18)] (https://arxiv.org/abs/1812.04767).  
The script is originally forked from [this repository] (https://github.com/rohanchandra30/Spectral-Trajectory-and-Behavior-Prediction).  
The script has been further revised in the network architecture, and it is still working on. 

## Depencencies:
* Python 3.5.3
* torch 1.3.1
* pytorch-ignite
* numpy 1.16.2

## How to generate dataset:
In this script, it uses apolloscape [dataset] (http://apolloscape.auto/trajectory.html). The datasets can be preprocessed as people, vehicles and bicycles/motorcycles. It means if you want to train the network for predicting people only, change 'CLASS_TYPE' to 'human' in **`traphic_main.py`**.  Change 'GENERATE_DATASET' as 'True', and run `traphic_main.py`.

## How to train:
1. Clone this repository
```
$ git clone <repository>
```

2. Training
```
$ python3 traphic_main.py
```


