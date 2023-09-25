
import os
import sys, re
import argparse

def write_list_to_txt4(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')


def filter_python_tests(target_files):
    filtered = []
    for f in target_files:
        if re.findall(r'tests\/', f) or re.findall(r'\/tests\/', f) or re.findall(r'\_test\.py', f):
            if f.endswith('.py'):
                filtered.append(f)
    return filtered


def getListOfFiles(dirName):
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

def main(args):
    #target_path = '/media//DATA/vsprojects/FSE23_2/data/tf/tf_test_files/tf_tests_v2.4.0.txt'
    #tf_root_path = 'path/to/tensorflow/cloned/dir'
    tf_root_path = args.tf_root_path
    output = args.output
    if not os.path.isfile(tf_root_path):
        # get list of all files
        _files = getListOfFiles(tf_root_path)
        # output the list to disk
        _files = filter_python_tests(_files)
        for f in _files:
            write_list_to_txt4(f, output)
        

if __name__ == '__main__':
    Epilog  = """python get_tf_test_files.py --tf_root_path=path/to/tensorflow/python/dir --output=/path/to/output"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Collect all test files from TensorFlow Python directory.', epilog=Epilog)
    
    parser.add_argument('--tf_version', type=str, default="v2.9", help='.')
    parser.add_argument('--tf_root_path' , type=str, help='Please enter the root directory of TensorFlow where high-level python modules are reside.')
    parser.add_argument('--output', type=str, help='Please enter the output directory where you want to save the list of TensorFlow test files.')
    
    args = parser.parse_args()

    if args.tf_root_path == None or args.output == None:
        parser.print_help()
        sys.exit(-1)
    
    main(args)