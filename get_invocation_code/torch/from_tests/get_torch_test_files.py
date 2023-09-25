
import os 
import re

ROOT_DIR = 'media//SSD/FSE23_2'

TORCH_BASE_PATH = '/media//SSD/pytorch/test'

def write_list_to_txt(data):
    fname = os.path.join('/media//SSD/FSE23_2/data/torch/torch_test_files/torch_tests.txt')
    with open(fname, "a") as file:
        #for row in data:
        file.write(str(data)+'\n')

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def get_test_files():
    _files = getListOfFiles(TORCH_BASE_PATH)
    
    for f in _files:
        f_split = f.split('/')
        if re.findall(r'test\_', f_split[-1]) and f_split[-1].endswith('.py'):
            write_list_to_txt(f)   

if __name__ == '__main__':
    get_test_files()