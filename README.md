## Preface
This is the prototype implementation of our paper namely "History-Driven Fuzzing for Deep Learning Libraries" submitted to the FSE2024 conference. Please note that we have taken careful steps to protect the anonymity of the replication package.

## Required Dependencies
The following are the dependencies required to run Orion:

```
inspect
pymongo
colorama
pandas
```

You can simply install all the required dependencies using the ```pip``` package manager.

## Running Orion
### Step 1 
In order to run Orion, first you need to install the target DL library. For example, if you want to test Orion on PyTorch-1.13.1, you need to run the following command:

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Please refer to the TensorFlow release history and PyTorch release history for further information.

### Step 2
In the second step, you will need to download and extract the source in your desired directory.

### Step 2
When you download the source code, you need to enter the following command in the terminal:
```
cd /root/fuzzing/
```

Then you have to enter the following command:

```
python run_fuzzer.py --database="database name" --library="target library" --release="target release" --tool="orion" --experiment_round=1
```

Example:

```
python run_fuzzer.py --database="orion-tf1" --library="tf" --release="2.11.0" --tool="orion" --experiment_round=1
```

## Data
The reported vulnerabilities are available for [TensorFlow](https://github.com/dmc1778/Orion/blob/master/ORION_Confirmed_TensorFlow_Vulnerabilities.csv) and [PyTorch](https://github.com/dmc1778/Orion/blob/master/ORION_Confirmed_Torch_Vulnerabilities.csv).
