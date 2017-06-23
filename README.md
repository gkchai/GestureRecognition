# About

Repository for building ML workflows for Gesture Recognition. The following workflows are supported: training,
evaluation, inference  and exporting.


## Application

3-axis accelerometer data from a smartphone is used to infer four possible gestures: pickup, steady, dropoff and unknown.


## Dataset

500 gestures signatures are collected. The time window is 2 sec (or 40 samples at 20 Hz sampling rate). Size of the
inputs or the number of features is 120 per gesture.

```
raw data from data collection:  ./raw_data/
processed data merged into train and test data: ./processed_data/
data in TF records format: ./dataset/
```


## Prerequisites

### Install dependencies
Preferred way is in virtualenv (Python 2.7)
```sh
cd GestureRecognition
pip install --upgrade pip
pip install -r requirements.txt
```


## Using the repo

### Structure
```
./train_dir/: Location of model checkpoints during or after training
./export_dir/: Location of exported protobuf files ready for production
./plots/: Location of generated plots

```


### Commands

Training:

```sh
$ python train.py
```

To evaluate an existing or newly generated model (can also be run in parallel during training):

```sh
$  python evaluate.py
```

To run inference on generated model:

```sh
$  python inference.py
```

To export saved graph and model checkpoint into a protobuf:

```sh
$  python export.py
```



## Results

Running inference on test data on a model trained for 10000 steps:

```
---- Metrics ------
             precision    recall  f1-score   support

     pickup       0.97      0.95      0.96        38
     steady       0.93      1.00      0.96        37
    dropoff       0.95      0.95      0.95        37
    unknown       0.94      0.89      0.92        38

avg / total       0.95      0.95      0.95       150
```

