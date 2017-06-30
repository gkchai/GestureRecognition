# About

Repository for building ML workflows for Gesture Recognition. The following workflows are supported: training,
evaluation, inference  and exporting. The scripts follow the design of well-known ML models 
(`https://github.com/tensorflow/models`) that have been implemented using TF-slim. Following deep models are 
 available: MLP, LSTM and CNN.

## Application

3-axis accelerometer data from a smartphone is used to classify four possible gestures: pickup, steady, dropoff and unknown.


## Dataset

500 gestures signatures are collected. The time window is 2 sec (or 40 samples at 20 Hz sampling rate). Total size of the
inputs (or the number of features) is 120 per gesture.

```
raw data from experiments:  ./raw_data/
processed data merged into train and test data: ./processed_data/
data in TF records format: ./dataset/
```


## Prerequisites

### Install dependencies
Preferred way is in virtualenv (Python 2.7/3.5)
```sh
pip install --upgrade pip
pip install -r requirements.txt
```


## Using the repo

### Structure
```
./train_dir/MLP: Location of MLP model checkpoints during or after training
./export_dir/: Location of exported protobuf files ready for production
./plots/: Location of generated plots

```


### Commands

Help:

```sh
$ python train.py --h
```


Training:

```sh
$ python train.py --model MLP --num_of_steps=20000
```

To evaluate an existing or newly generated model ( run in parallel in another window during training):

```sh
$  python evaluate.py  --model MLP
```

To run inference on generated model:

```sh
$  python inference.py  --model MLP
```

To export saved graph and model checkpoint into a protobuf (run inference before exporting):

```sh
$  python export.py  --model MLP
```

## Summary

```sh
$  tensorboard --logdir=/tmp/ges_rec_logs/
```


## Results

Running inference on an CNN model trained for 10000 steps:

```sh
$  python inference.py  --model CNN
```


```
---- Metrics ------
             precision    recall  f1-score   support

     pickup       1.00      0.92      0.96        38
     steady       0.90      1.00      0.95        37
    dropoff       0.95      0.95      0.95        37
    unknown       0.95      0.92      0.93        38

avg / total       0.95      0.95      0.95       150

---- Confusion Matrix ------
[[35  2  0  1]
 [ 0 37  0  0]
 [ 0  1 35  1]
 [ 0  1  2 35]]
```

