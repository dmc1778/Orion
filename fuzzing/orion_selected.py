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
from classes.argument import Argument
from classes.database import TorchDatabase, TFDatabase
import re
import copy
import os
import random

partitions = {
    "<ArgType.INT: 1>": {
        'NEGATIVE_INTEGERS': '',
        'ZERO_INTEGER': '',
        'LARGE_INTEGER': '[10000, 10000000, 5000000]',
        'NEGATIVE_LARGE_INTEGER': '',
        'EMPTY': '',
        'NAN': '',
        'NONE': ''

    },
    "<ArgType.FLOAT: 3>": {
        'NEGATIVE_FLOAT': '',
        'ZERO_FLOAT': '',
        'LARGE_FLOAT': '',
        'NEGATIVE_LARGE_FLOAT': '',
        'EMPTY': '',
        'NAN': '',
        'NONE': ''
    },
    '<ArgType.STR: 2>':
    {
        "INVALID_STRING": '',
        "EMPTY_STRING": '',
                        'EMPTY': '',
                        'NAN': '',
                        'NONE': ''
    },
    '<ArgType.LIST: 6>': {
        'LARGE_INTEGER_LIST_ELEMENT': '',
        'ZERO_INTEGER_LIST_ELEMENT': '',
        'NEGATIVE_INTEGER_LIST_ELEMENT': '',
        'EMPTY_INTEGER_LIST_ELEMENT': '',
        'INVALID_STRING_LIST_ELEMENT': '',
        'NAN_INPUT_LIST_ELEMENT': '',
        'NONE_INPUT_LIST_ELEMENT': '',
    },
    '<ArgType.TUPLE: 5>': {
        'LARGE_INTEGER_TUPLE_ELEMENT': '',
        'ZERO_INTEGER_TUPLE_ELEMENT': '',
        'NEGATIVE_INTEGER_TUPLE_ELEMENT': '',
        'EMPTY_INTEGER_TUPLE_ELEMENT': '',
        'INVALID_STRING_TUPLE_ELEMENT': '',
        'NAN_INPUT_TUPLE_ELEMENT': '',
        'NONE_INPUT_TUPLE_ELEMENT': '',
    },
    '<ArgType.BOOL: 4>': {
        'NONE': '',
        'NAN': '',
        'EMPTY': '',
        'ZERO': '',
    },
    '<ArgType.TORCH_TENSOR: 9>': {
        # "NON_SCALAR_INPUT_TENSOR": '',
        "LARGE_INPUT_TENSOR_1": '',
        'LARGE_INPUT_TENSOR_2': '',
        'NEGATIVE_INPUT_TENSOR': '',
        'SCALAR_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR_WHOLE': '',
        'TENSOR_WITH_ZERO_ELEMENT_1': '',
        'TENSOR_WITH_ZERO_ELEMENT_2': '',
        'TENSOR_EMPTY_FLAG_1': '',
    },
    '<ArgType.TF_TENSOR: 11>': {
        # "NON_SCALAR_INPUT_TENSOR": '',
        "LARGE_INPUT_TENSOR_1": '',
        'LARGE_INPUT_TENSOR_2': '',
        'NEGATIVE_INPUT_TENSOR': '',
        'SCALAR_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR': '',
        'NAN_INPUT_TENSOR_WHOLE': '',
        'TENSOR_WITH_ZERO_ELEMENT_1': '',
        'TENSOR_WITH_ZERO_ELEMENT_2': '',
        'TENSOR_EMPTY_FLAG_1': '',
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

    library = "torch"

    output_dir = "/media/nimashiri/SSD/testing_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    rules = [
        "MUTATE_PREEMPTIVES",
        "NEGATE_INT_TENSOR",
        "RANK_REDUCTION_EXPANSION",
        "EMPTY_TENSOR_TYPE1",
        "EMPTY_TENSOR_TYPE2",
        "EMPTY_LIST",
        "ZERO_TENSOR_TYPE1",
        "ZERO_TENSOR_TYPE2",
        "NAN_TENSOR",
        "NAN_TENSOR_WHOLE",
        "NON_SCALAR_INPUT",
        "SCALAR_INPUT",
        "LARGE_TENSOR_TYPE1",
        "LARGE_TENSOR_TYPE2",
        "LARGE_LIST_ELEMENT",
    ]

    buggy_api = "/media/nimashiri/SSD/testing_results/runcrash.txt"
    if library == "tf":

        MyTF = TFLibrary(output_dir)
        TFDatabase.database_config("localhost", 27017, "freefuzz-tf")

        try:
            for api_name in TFDatabase.get_api_list():
                api = TFAPI(api_name)
                for arg in api.args:
                    original_arg = copy.copy(api.args[arg])
                    _type = repr(api.args[arg].type)
                    sub_partitions = partitions[_type]
                    for part in sub_partitions:
                        for itr in range(50):
                            print(
                                '###############################################################################################')
                            print(
                                f"Running on new component API::{api_name}, Argument::{arg} Partition::{part} Iteration::{itr}")
                            print(
                                '###############################################################################################')
                            api.each_arg_mutate(api.args[arg], part)
                            MyTF.test_with_oracle(api, OracleType.CRASH)
                            api.api = api_name
                            MyTF.test_with_oracle(api, OracleType.CUDA)
                            api.api = api_name
                            api.args[arg] = original_arg
        except Exception as e:
            print(e)

        try:
            for api_name in TFDatabase.get_api_list():
                for k in range(1):
                    # old_api = copy.deepcopy(api)

                    api_keywords = api_name.split(".")
                    if api_keywords.count("tensorflow") > 1:
                        api_name = make_api_name_unique(api_name)

                    api = TFAPI(api_name)
                    for c1 in range(1000):
                        print(
                            "########################################################################################################################"
                        )
                        print(
                            f"The current API under test::{api_name}. Working on dimension mismatch, Iteration_L1 {k}, Iteration_L2 {c1}")
                        print(
                            "########################################################################################################################"
                        )
                        api.new_mutate_tf()

                        MyTF.test_with_oracle(api, OracleType.CRASH)
                        api.api = api_name

                        MyTF.test_with_oracle(api, OracleType.CUDA)
                        api.api = api_name
        except Exception as e:
            print(e)

        # try:
        #     for api_name in TFDatabase.get_api_list():

        #         for k in range(1):
        #             api_keywords = api_name.split(".")
        #             if api_keywords.count("tensorflow") > 1:
        #                 api_name = make_api_name_unique(api_name)

        #             for j in range(1000):
        #                 old_api = TFAPI(api_name)
        #                 for i, arg in enumerate(old_api.args):
        #                     for r in rules:
        #                         print(
        #                             "########################################################################################################################"
        #                         )
        #                         print(
        #                             "The current API under test: ###{0}###. Mutating the parameter ###{1}### using the rule ###{2}###, Iteration: {3}".format(
        #                                 api_name, arg, r, j
        #                             )
        #                         )
        #                         print(
        #                             "########################################################################################################################"
        #                         )
        #                         old_arg = copy.deepcopy(old_api.args[arg])
        #                         old_api.new_mutate_multiple(
        #                             old_api.args[arg], r)
        #                         MyTF.test_with_oracle(
        #                             old_api, OracleType.CRASH)
        #                         old_api.api = api_name
        #                         MyTF.test_with_oracle(old_api, OracleType.CUDA)
        #                         # api.api = sys.argv[2]
        #                         # MyTF.test_with_oracle(api, OracleType.PRECISION)
        #                         old_api.api = api_name
        #                         old_api.args[arg] = old_arg
        # except Exception as e:
        #     print(e)

    else:
        import torch
        from classes.torch_library import TorchLibrary
        from classes.torch_api import TorchAPI
        from classes.database import TorchDatabase

        TorchDatabase.database_config("localhost", 27017, "freefuzz-torch")

        MyTorch = TorchLibrary(output_dir)

        for api_name in TorchDatabase.get_api_list():
            # torch.einsum has string
            # torch.cummin has list

            api = TorchAPI(api_name)
            for arg in api.args:
                original_arg = copy.copy(api.args[arg])
                _type = repr(api.args[arg].type)
                sub_partitions = partitions[_type]
                for part in sub_partitions:
                    for itr in range(50):
                        print(
                            '###############################################################################################')
                        print(
                            f"Running on new component API::{api_name}, Argument::{arg} Partition::{part} Iteration::{itr}")
                        print(
                            '###############################################################################################')
                        api.each_arg_mutate(api.args[arg], part)
                        MyTorch.test_with_oracle(api, OracleType.CRASH)
                        api.api = api_name
                        MyTorch.test_with_oracle(api, OracleType.CUDA)
                        api.api = api_name
                        api.args[arg] = original_arg

            for k in range(1):
                api = TorchAPI(api_name)
                for c1 in range(1000):

                    print(
                        "########################################################################################################################"
                    )
                    print(
                        f"The current API under test::{api_name}. Working on dimension mismatch, Iteration_L1 {k}, Iteration_L2 {c1}")
                    print(
                        "########################################################################################################################"
                    )
                    api.new_mutate_torch()

                    MyTorch.test_with_oracle(api, OracleType.CRASH)
                    api.api = api_name
                    MyTorch.test_with_oracle(api, OracleType.CUDA)
                    api.api = api_name
