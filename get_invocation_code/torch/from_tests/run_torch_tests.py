import subprocess, os
from multiprocessing import Pool
from pymongo import MongoClient
import multiprocessing, threading
from threading import Thread
import logging, re
from pymongo.errors import ServerSelectionTimeoutError

logging.basicConfig(level = logging.INFO)

def read_txt(fname):
    with open(fname, 'r') as fileReader:
        data = fileReader.read().splitlines()
    return data

def process_prerun(input_addr):
    print("executed {}. thread".format(input_addr))


def process_prerun(input_addr):
    print("executed {}. thread".format(input_addr))

def write_list_to_txt4(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')

class KillableThread(Thread):
    def __init__(self, sleep_interval, input_addr):
        super().__init__()
        self._kill = threading.Event()
        self._interval = sleep_interval
        self.input_addr = input_addr

    def run(self):
        while True:
            process_prerun(self.input_addr)

            # If no kill signal is set, sleep for the interval,
            # If kill signal comes in while sleeping, immediately
            #  wake up and handle
            is_killed = self._kill.wait(self._interval)
            if is_killed:
                break

        print("Killing Thread")

    def kill(self):
        self._kill.set()

def process_prerun(input_addr):
    print("Executed:#: {}.".format(input_addr))

    command = 'python3 '+input_addr
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

class RunTestFiles():
    def __init__(self, test_files) -> None:
        self.test_files = test_files

    def callback(self, e):
        print('I am in callback!')
        self.event.wait()

        if e is not None:
            self.test_files_status[e[0]] = e[1]

    def runProcess(self, exe):
        p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while(True):
            retcode = p.poll()
            line = p.stdout.readline()
            yield line
            if retcode is not None:
                break

    def pre_run_test_files(self):
        with Pool(10) as p:
            p.map(process_prerun, self.test_files)


def run_tf_tests(data):
    _path_clean_tests = '/media//SSD/FSE23_2/data/torch/torch_test_files/history.txt'
    _path_corrupted_tests = '/media//SSD/FSE23_2/data/torch/torch_test_files/corrupted.txt'

    mode = 'a' if os.path.exists(_path_clean_tests) else 'w'
    f1 = open(_path_clean_tests, mode=mode)

    mode = 'a' if os.path.exists(_path_corrupted_tests) else 'w'
    f2 = open(_path_corrupted_tests, mode=mode)

    hist = read_txt(_path_clean_tests)
    corr = read_txt(_path_corrupted_tests)

    for i, t in enumerate(data):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f'Running the test {t}:::index{i}')
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=10, connectTimeoutMS=300)
        try:
            info = client.server_info()
        except ServerSelectionTimeoutError:
                logging.info('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                logging.info('#### MongoDB Server is Down! I am trying initiating the server now. ####')
                logging.info('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                
                subprocess.call("sudo systemctl restart mongod", shell=True)
                
        if t not in hist and t not in corr:
            try:
                subprocess.run(['python3', t], timeout=150)
                write_list_to_txt4(t, _path_clean_tests)
            except subprocess.TimeoutExpired:
                write_list_to_txt4(t, _path_corrupted_tests)
        else:
            print("This test already parsed!")

if __name__ == '__main__':

    tf_tests = '/media//SSD/FSE23_2/data/torch/torch_test_files/torch_tests.txt'
    data = read_txt(tf_tests)
    run_tf_tests(data)

