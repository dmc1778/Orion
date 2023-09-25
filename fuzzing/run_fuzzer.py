from utils.printer import dump_data
import re, shutil
import subprocess
from classes.database import TorchDatabase
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import logging
import sys
import os
import traceback
from constants.enum import OracleType
from os.path import join
from utils.converter import str_to_bool
import tensorflow as tf
import argparse
import pandas as pd
from colorama import init, Fore,  Style
# Instantiate the parser
parser = argparse.ArgumentParser(description='Fuzzing DL libraries')
# from classes.tf_library import TFLibrary
# from classes.tf_api import TFAPI

# from classes.database import FDatabase

logging.basicConfig(level=logging.INFO)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")


def read_txt(fname):
    with open(fname, "r") as fileReader:
        data = fileReader.read().splitlines()
    return data


def write_list_to_txt4(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')


def write_to_dir(dir, api_name, code):
    api_dir = join(dir, api_name)
    if not os.path.exists(api_dir):
        os.makedirs(api_dir)
    filenames = os.listdir(api_dir)
    max_name = 0
    for name in filenames:
        max_name = max(max_name, int(name.replace(".py", "")))
    new_name = str(max_name + 1)
    with open(join(api_dir, new_name + ".py"), "w") as f:
        f.write(code)


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


def find_skip_list(api_):
    flag_ = True
    skip_list = [
        "python.autograph",
        "python.debug",
        "python.distribute",
        "python.eager",
        "python.framework",
        "python.grappler",
    ]
    if "tf." in api_:
        return False
    elif "torch." in api_:
        return True
    else:
        split_api = api_.split(".")
        target = ".".join([split_api[1], split_api[2]])
        if target in skip_list:
            flag_ = False
    return flag_


class InterpreterError(Exception):
    pass


def my_exec(cmd, globals=None, locals=None, description="source string"):
    existence_flag = False
    try:
        exec(cmd, globals, locals)
    except SyntaxError as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        if "SyntaxError" == error_class:
            existence_flag = True
        line_number = err.lineno
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        if "AttributeError" == error_class or "ModuleNotFoundError" == error_class:
            existence_flag = True
        cl, exc, tb = sys.exc_info()
        line_number = traceback.extract_tb(tb)[-1][1]
    else:
        return existence_flag
    return existence_flag


def pre_run_check(api_):
    code = "import tensorflow as tf\n"
    if re.findall(r"(tensorflow\.python)", api_):
        part_from = ".".join(api_.split(".")[0:-2])
        code += f"from {part_from} import {api_.split('.')[-2]}\n"
        api_ = ".".join(api_.split(".")[-2:])
        flag = my_exec(code)

    else:
        code += api_
        flag = my_exec(code)
    return flag


def search_in_dataset(api_name, lib):
    data = read_txt('/media//SSD/FSE23_2/data/benchmarking/tf_test_apis.txt')
    
    flag = False
    if api_name in data:
        flag = True
    # if lib == 'torch':
    #     data = pd.read_csv(
    #         '/media//SSD/FSE23_2/data/benchmarking/torch_data.csv')
    # else:
    #     data = pd.read_csv(
    #         '/media//SSD/FSE23_2/data/benchmarking/tf_data_fse24.csv')
    # for idx, row in data.iterrows():
    #     if api_name == row['API Name']:
    #         flag = True
    return flag


def run_fuzzer(args):
    # dbname = "freefuzz-tf"
    # library = "tf_new"
    # tool = "orion"
    # round_exp = 1
    # config_name = ''

    # component1 = 'True'
    # component2 = 'True'
    # component3 = 'True'
    # release = "2.4.0"
    
    dbname = args.database
    library = args.library
    release = args.release
    tool = args.tool
    round_exp = args.experiment_round
    config_name = args.config_dir
    
    

    #for numRun in range(5):
    #library = f"{library}_{numRun}"
    tf_output_dir = f"/media//SSD/testing_results/{tool}/{library}/{release}"

    if not os.path.exists(tf_output_dir):
        os.makedirs(tf_output_dir, exist_ok=True)

    mydb = myclient[dbname]
        #TorchDatabase.database_config("localhost", 27017, dbname)
    config_name = "/media//SSD/FSE23_2/fuzzing/config/expr.conf"

    if round_exp == 1:
        data = mydb.list_collection_names()
    else:
        buggy_api = f"{tf_output_dir}/runcrash.txt"
        data = read_txt(buggy_api)

    hisotry_file = f'{tool}_{library}_{release}_executed_apis.txt'

    if not os.path.exists(hisotry_file):
        f1 = open(hisotry_file, 'a')

    hist = read_txt(f'{tool}_{library}_{release}_executed_apis.txt')
    components = ['component1', 'component2', 'component3']
        
    for i, api_ in enumerate(data):
        if api_ not in hist:
            #api_ = "tensorflow.python.eager.lift_to_graph.lift_to_graph"
            write_list_to_txt4(api_, f'{tool}_{library}_{release}_executed_apis.txt')
            flag = True
            # flag = search_in_dataset(api_, library)
            if flag:        
                for component in components:
                    print(
                            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print(f"Running API {api_}::{i}/{len(data)} ON {component}")
                    print(
                            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # if 'tensorflow.python' in api_:
                    #     prerun_flag = pre_run_check(api_)
                    # else:
                    #     prerun_flag = False
                    # prerun_flag = True
                    
                    
                        # skip_flag = find_skip_list(api_)
                    if "__main__" not in api_:
                            try:
                                if tool == "orion":
                                    res = subprocess.run(
                                        [
                                            "python3",
                                            "/media//SSD/FSE23_2/fuzzing/orion_main.py",
                                            library,
                                            api_,
                                            str(i),
                                            tool,
                                            dbname,
                                            tf_output_dir,
                                            component,
                                        ],
                                        capture_output=True,
                                        text=True,
                                        shell=False,
                                        timeout=100,
                                    )
                                    print('sdsdsd')
                                elif tool == "FreeFuzz":
                                    res = subprocess.run(
                                            [
                                                "python3",
                                                "/media//SSD/FSE23_2/fuzzing/freefuzz_api_main.py",
                                                config_name,
                                                library,
                                                api_,
                                                str(i),
                                                tool,
                                            ],
                                            shell=False,
                                            timeout=100,
                                        )
                                else:
                                    print("No tool provided")
                            except subprocess.TimeoutExpired:
                                dump_data(f"{api_}\n", join(
                                    tf_output_dir, "timeout.txt"), "a")
                            except Exception as e:
                                dump_data(
                                    f"{api_}\n  {e}\n", join(
                                            tf_output_dir, "runerror.txt"), "a"
                                    )
                            else:
                                if res.returncode != 0:
                                    if "AttributeError: module" not in res.stderr and "NameError:" not in res.stderr:
                                        print(f"{Fore.RED} I found a potential bug: {res.stderr}{Style.RESET_ALL}")
                                        if round_exp == 1:
                                            dump_data(f"{api_}\n", join(
                                                tf_output_dir, "runcrash.txt"), "a")
                                        else:
                                            dump_data(f"{api_}\n", join(
                                                tf_output_dir, "runcrashround2.txt"), "a")

                                        buggy_path = f"{tf_output_dir}/{tool}_bugs"

                                        if not os.path.exists(buggy_path):
                                            os.makedirs(buggy_path, exist_ok=True)

                                        command_ = f"cp -r {tf_output_dir}/temp.py {buggy_path}/{api_}.py"
                                            
                                        command_2 = f"cp -r {tf_output_dir}/rule_temp.txt {buggy_path}/{api_}_output.txt"
                                        
                                        subprocess.call(command_2, shell=True)
                                            
                                        try:
                                            with open(f'{buggy_path}/{api_}_error.txt', 'w') as f:
                                                f.write(res.stderr)
                                        except Exception as e:
                                            print('Could not write the std error!')
                                            #command_ = f"find {tf_output_dir} -name '*.py' -exec cp {{}} {buggy_path} \;"
                                            
                                        subprocess.call(command_, shell=True)
                                            
                                            #subprocess.call(f'find {tf_output_dir} -name "*.py" -exec rm {{}} \;', shell=True)
                                    else:
                                        print(f"{Fore.GREEN} I found a: {res.stderr}{Style.RESET_ALL}")
                    else:
                        print("API Skipped!")
            else:
                print('This API is not in our target API list!')
        else:
            print('This api has been already tested!')

def str_to_bool_list(value):
    bool_list = value.split(',')
    if len(bool_list) != 3:
        raise argparse.ArgumentTypeError('Three boolean values expected.')
    return [bool(x) for x in bool_list]


if __name__ == "__main__":
    Epilog = """An example usage: python run_fuzzer.py --database="freefuzz-tf" --library="tf" --release="2.11.0" --tool="orion" --experiment_round=1"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Extract commits from github repositories.', epilog=Epilog)

    parser.add_argument('--database', type=str, default="orion-tf1",
                        help='Please enter the name of the database.')
    parser.add_argument('--library', type=str, default="tf",
                        help='Please enter the library name.')
    parser.add_argument('--release', type=str, default="2.13.0",
                        help='Please enter the library name.')
    parser.add_argument('--tool', type=str, default="orion",
                        help='Please enter the name of the tool')
    parser.add_argument('--experiment_round', default=1,
                        type=int, help='Please specify the round of experiment.')

    parser.add_argument('--config_dir', default=1,
                        type=str, help='Please specify the dir for configurations.')

    args = parser.parse_args()

    if args.database == None or args.library == None or args.tool == None:
        parser.print_help()
        sys.exit(-1)

    
    run_fuzzer(args)
    # run_fuzzer()
