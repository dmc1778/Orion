import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import logging
import sys
from constants.enum import OracleType
from os.path import join
from utils.converter import str_to_bool
import tensorflow as tf
from classes.tf_library import TFLibrary
from classes.tf_api import TFAPI
from classes.torch_api import TorchArgument
from classes.tf_api import TFArgument 
from classes.database import TorchDatabase, TFDatabase
import re
import copy
import os
import random

partitions = {
    "<ArgType.INT: 1>": {
        'NEGATIVE_INTEGERS': '',
        'ZERO_INTEGER':'',
        'LARGE_INTEGER':'',
        'NEGATIVE_LARGE_INTEGER':'',
        'EMPTY':'',
        'NONE':'',
        'NAN':'',
    },
    "<ArgType.NULL: 7>":{
        'NULL':''
    },
    "<ArgType.FLOAT: 3>": {
        'NEGATIVE_FLOAT':'',
        'ZERO_FLOAT':'',
        'LARGE_FLOAT':'',
        'NEGATIVE_LARGE_FLOAT':'',
        'EMPTY':'',
        'NONE':'',
        'NAN':'',
    },
    '<ArgType.STR: 2>':
    {
        "INVALID_STRING": '',
        "EMPTY_STRING": '',
        'EMPTY_ASSIGNMENT': '',
        'NAN': '',
        'NONE': ''
    },
    '<ArgType.LIST: 6>': {
        'LARGE_LIST_ELEMENT':"",
        'NEGATIVE_LARGE_LIST_ELEMENT': '',
        'ZERO_LIST_ELEMENT': '',
        'NEGATIVE_LIST_ELEMENT': '',
        'EMPTY_LIST_ELEMENT': '',
        'INVALID_LIST_ELEMENT': '',
        'NONE_INPUT_LIST_ELEMENT': '',
        'NAN_INPUT_LIST_ELEMENT':'',
    },
    '<ArgType.TUPLE: 5>': {
        'LARGE_TUPLE_ELEMENT': '',
        'ZERO_TUPLE_ELEMENT': '',
        'NEGATIVE_TUPLE_ELEMENT': '',
        'EMPTY_TUPLE_ELEMENT': '',
        'INVALID_TUPLE_ELEMENT': '',
        'NONE_INPUT_TUPLE_ELEMENT': '',
        'NAN_INPUT_TUPLE_ELEMENT': '',
    },
    '<ArgType.BOOL: 4>': {
        'RANDOM_BOOL': '',
        'NONE_BOOL': '',
        'NAN_BOOL': '',
        'EMPTY_BOOL': '',
        'ZERO_BOOL': '',
    },
    '<ArgType.TORCH_TENSOR: 9>': {
        "NON_SCALAR_INPUT_TENSOR": '',
        "LARGE_INPUT_TENSOR": '',
        'NEGATIVE_INPUT_TENSOR': '',
        'SCALAR_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR_WHOLE': '',
        'TENSOR_EMPTY_FLAG': '',
    },
    '<ArgType.TF_TENSOR: 11>': {
        "NON_SCALAR_INPUT_TENSOR": '',
        "LARGE_INPUT_TENSOR": '',
        'NEGATIVE_INPUT_TENSOR': '',
        'SCALAR_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR_WHOLE': '',
        'TENSOR_EMPTY_FLAG': '',
    },
    "<ArgType.TF_OBJECT: 15>":{
        'NULL_TF_OBJ':''
    },
    "<ArgType.TORCH_OBJECT: 8>":{
        'NULL_TF_OBJ':''
    },
    "<ArgType.TORCH_DTYPE: 10>":{
        'NULL_TF_OBJ':''
    },
    "<ArgType.TF_DTYPE: 12>":{
        'NULL_TF_OBJ':''
    }
}


logging.basicConfig(level=logging.INFO)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")


def read_txt(fname):
    with open(fname, "r") as fileReader:
        data = fileReader.read().splitlines()
    return data


def check_connection():
    client = MongoClient(
        "mongodb://localhost:27017/", serverSelectionTimeoutMS=10, connectTimeoutMS=300
    )

    try:
        info = client.server_info()

    except ServerSelectionTimeoutError:
        logging.info(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )
        logging.info(
            "#### MongoDB Server is Down! I am trying initiating the server now. ####"
        )
        logging.info(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )


def make_api_name_unique(api):
    api_split = api.split("tensorflow")
    new_api_name = "tensorflow" + api_split[-1]
    return new_api_name


def count_tensor_inputs(api, lib="Tensorflow"):
    tensor_holder = []
    integer_holder = []
    for arg in api.args:
        _arg = api.args[arg]

        if lib == "Tensorflow":
            if re.findall(r"(ArgType\.TF\_TENSOR\:)", repr(_arg.type)):
                tensor_holder.append(1)
            if re.findall(r"ArgType\.INT\:", repr(_arg.type)):
                integer_holder.append(1)
        else:
            if re.findall("r(ArgType\.TORCH\_TENSOR\:)", repr(_arg.type)):
                tensor_holder.append(1)
    return tensor_holder

if __name__ == "__main__":
    # library = sys.argv[1]
    # api_name = sys.argv[2]
    # index = sys.argv[3]
    # tool_name = sys.argv[4]
    # dbname = sys.argv[5]
    # output_dir = sys.argv[6]
    # component = sys.argv[7]
    # print('sds1')

    library = "tf"
    api_name = "tensorflow.python.ops.gen_array_ops.lower_bound"
    index = 1
    tool_name = "orion"
    dbname = 'orion-tf1'
    release = "2.13.0"
    output_dir = f"/media//SSD/testing_results/{tool_name}/{library}/{release}"
    component = "component2"
    

    # output_dir = "/media//SSD/testing_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rules = [
        "LARGE_INTEGER",
        "NEGATIVE_INTEGER",
        "NEGATIVE_LARGE_INTEGER",
        "ZERO_INTEGER",
        "EMPTY_INTEGER",
        "NAN_INTEGER",
        "NONE_INTEGER",
        "LARGE_FLOAT",
        "NEGATIVE_FLOAT",
        "NEGATIVE_LARGE_FLOAT",
        "ZERO_FLOAT",
        "EMPTY_FLOAT",
        "NAN_FLOAT",
        "NONE_FLOAT",
        "INVALID_STRING",
        "EMPTY_STRING1",
        "EMPTY_STRING2",
        "NAN_STRING",
        "NONE_STRING",
        "RANK_REDUCTION_EXPANSION",
        "EMPTY_TENSOR_TYPE1",
        "EMPTY_TENSOR_TYPE2",
        "EMPTY_LIST",
        "LARGE_TENSOR_TYPE1",
        "LARGE_TENSOR_TYPE2",
        "LARGE_LIST_ELEMENT",
        "ZERO_TENSOR_TYPE1",
        "ZERO_TENSOR_TYPE2",
        "NAN_TENSOR",
        "NAN_TENSOR_WHOLE",
        "NON_SCALAR_INPUT",
        "SCALAR_INPUT"]

    if "tf" in library or "tf_new" in library:

        MyTF = TFLibrary(output_dir)
        TFDatabase.database_config("localhost", 27017, dbname)
        
        if component == "component1":
                com = "com1"
                print(f"I am here at{com}")
                for itr in range(1000):
                    api = TFAPI(api_name)
                    original_api = copy.copy(api)
                    for arg in api.args:
                        _type = repr(api.args[arg].type)	
                        sub_partitions = partitions[_type]
                        for part in sub_partitions:
                            print('#######################################################################################################')
                            print(f"Running on component {com}####API::{api_name}####Argument::{arg}####Partition::{part} Iteration::{itr}")
                            print('#######################################################################################################')
                            api.each_arg_mutate(api.args[arg], part)

                            with open(f'{output_dir}/rule_temp.txt', 'w') as f:         
                                f.write(",".join(map(str, ['M1', api_name, arg, part])))
                            MyTF.test_with_oracle(api, OracleType.CRASH)
                            api.api = api_name
                            MyTF.test_with_oracle(api, OracleType.CUDA)
                            api.api = api_name
                            api = original_api
        if component == 'component2':
            com = "com2"
            try:
                for k in range(1):
                    api_keywords = api_name.split(".")
                    if api_keywords.count("tensorflow") > 1:
                        api_name = make_api_name_unique(api_name)

                    api = TFAPI(api_name)
                    for c1 in range(1000):
                        print(
                            "########################################################################################################################"
                        )
                        print(
                            "Running {0} on the current API under test: {1}/Index: {2}. Working on dimension mismatch, Iteration_L1 {3}, Iteration_L2 {4}".format(
                                tool_name, api_name, index, k, c1
                            )
                        )
                        print(
                            "########################################################################################################################"
                        )
                        api.new_mutate_tf()

                        with open(f'{output_dir}/rule_temp.txt', 'w') as f:         
                            f.write(",".join(map(str, ['M2', api_name])))        
                        MyTF.test_with_oracle(api, OracleType.CRASH)
                        api.api = api_name

                        MyTF.test_with_oracle(api, OracleType.CUDA)
                        api.api = api_name
            except Exception as e:
                print(e)

        if component == 'component3':
            com = 'com3'
            try:
                for k in range(1):
                    api_keywords = api_name.split(".")
                    if api_keywords.count("tensorflow") > 1:
                        api_name = make_api_name_unique(api_name)
                    for c1 in range(1000):
                        api = TFAPI(api_name)
                        original_api = copy.copy(api)
                        num_arg = len(api.args)
                        num_Mutation = random.randint(1, num_arg + 1)
                        for _ in range(num_Mutation):
                            arg_name = random.choice(list(api.args.keys()))
                            arg = api.args[arg_name]
                            for r in rules:
                                print(
                                    "########################################################################################################################"
                                )
                                print(f"Running {tool_name} on ###{api_name}###. Index: {index} Mutating the parameter ###{arg_name}### using the rule ###{r}###, Iteration: {c1}")
                                print(
                                    "########################################################################################################################"
                                )
                                api.new_mutate_multiple(arg, r)
                                
                                with open(f'{output_dir}/rule_temp.txt', 'w') as f:
                                    f.write(",".join(map(str, ['M3', api_name, arg_name, r])))
                                MyTF.test_with_oracle(api, OracleType.CRASH)
                                api.api = api_name
                                MyTF.test_with_oracle(api, OracleType.CUDA)
                                api.api = api_name
                                api = original_api
            except Exception as e:
                pass
    else:
        from classes.torch_library import TorchLibrary
        from classes.torch_api import TorchAPI
        from classes.database import TorchDatabase

        TorchDatabase.database_config("localhost", 27017, dbname)

        MyTorch = TorchLibrary(output_dir)

        if component == 'component1':
            com = 'com1'
            for itr in range(1000):
                api = TorchAPI(api_name)
                original_api = copy.copy(api)
                for arg in api.args:
                    #original_arg = copy.copy(api.args[arg])
                    _type = repr(api.args[arg].type)
                    sub_partitions = partitions[_type]
                    for part in sub_partitions:
                        print(
                                    '###############################################################################################')
                        print(
                                    f"Running on component {com}####API::{api_name}####Argument::{arg}####Partition::{part} Iteration::{itr}")
                        print(
                                    '###############################################################################################')
                        api.each_arg_mutate(api.args[arg], part)

                        with open(f'{output_dir}/rule_temp.txt', 'w') as f:    
                            f.write(",".join(map(str, ['M1', api_name, arg, part])))       
                        MyTorch.test_with_oracle(api, OracleType.CRASH)
                        api.api = api_name
                        MyTorch.test_with_oracle(api, OracleType.CUDA)
                        api.api = api_name
                        #api.args[arg] = original_arg
                        api = original_api
                        
        if component == 'component2':
            com = 'com2'
            try:
                for k in range(1):
                    api = TorchAPI(api_name)
                    for c1 in range(1000):

                        print(
                            "########################################################################################################################"
                        )
                        print(
                            "Running {0} on the current API under test: {1}/Index: {2}. Working on dimension mismatch, Iteration_L1 {3}, Iteration_L2 {4}".format(
                                tool_name, api_name, index, k, c1
                            )
                        )
                        print(
                            "########################################################################################################################"
                        )
                        api.new_mutate_torch()
                        
                        with open(f'{output_dir}/rule_temp.txt', 'w') as f:
                            f.write(",".join(map(str, ['M2', api_name])))
                        MyTorch.test_with_oracle(api, OracleType.CRASH)
                        MyTorch.test_with_oracle(api, OracleType.CUDA)
            except Exception as e:
                print(e)

        if component == 'component3':
            com = 'com2'
            try:
                for k in range(1):
                    for c1 in range(1000):
                        api = TorchAPI(api_name)
                        original_api= copy.copy(api)
                        num_arg = len(api.args)
                        num_Mutation = random.randint(1, num_arg + 1)
                        for _ in range(num_Mutation):
                            arg_name = random.choice(list(api.args.keys()))
                            arg = api.args[arg_name]
                            for r in rules:
                                print(
                                    "########################################################################################################################"
                                )
                                print(f"Running {tool_name} on ###{api_name}###. Index: {index} Mutating the parameter ###{arg_name}### using the rule ###{r}###, Iteration: {c1}")
                                print(
                                    "########################################################################################################################"
                                )
                                api.new_mutate_multiple(arg, r)
                                
                                with open(f'{output_dir}/rule_temp.txt', 'w') as f:
                                    f.write(",".join(map(str, ['M3', api_name, arg_name, r])))
                                    
                                MyTorch.test_with_oracle(api, OracleType.CRASH)
                                MyTorch.test_with_oracle(api, OracleType.CUDA)
                                api = original_api
            except Exception as e:
                print(e)
