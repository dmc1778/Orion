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

## Running Orion
### Step 1
First, you will need to download and extract the source in your desired directory.

### Step 2
When you download the source code, you need to enter the following command in the terminal:
```
cd /root/fuzzing/
```

Then you have to enter the following command:

```
python run_fuzzer.py --database="database name" --library="target library" --release="target release" --tool="orion" --experiment_round=1
```

## Data
The reported vulnerabilities are available for [TensorFlow](https://github.com/dmc1778/Orion/blob/master/ORION_Confirmed_TensorFlow_Vulnerabilities.csv) and [PyTorch](https://github.com/dmc1778/Orion/blob/master/ORION_Confirmed_Torch_Vulnerabilities.csv).
