import os, glob

def read_txt(addr):
    with open(addr, mode='r') as f:
        lines = f.readlines()
    return lines

def main():
    log_dir = "/media/nimashiri/SSD/testing_results/DeepRel/tf/2.3.0/output-0/logs/test-confirmed-apis-2.txt"
    pair_dir = "/media/nimashiri/SSD/testing_results/DeepRel/tf/2.3.0/output-0/"

    match_ = read_txt(log_dir)
    all_pair_dir = os.listdir(pair_dir)
    counter = 0
    for file in match_:
        new_file = file.replace(" ", "+")
        new_file = f"{new_file}+2+ver"
        new_file_1 = new_file.replace("\n","")
        new_file_2 = new_file_1.replace("2","1")
        
        if new_file_1 in all_pair_dir or new_file_2 in all_pair_dir:
            counter = counter + 1
            print(counter)
            f1_ = os.path.join(pair_dir, new_file_1)
            f2_ = os.path.join(pair_dir, new_file_2)
            flag1 = os.path.exists(f1_)
            flag2 = os.path.exists(f2_)
            if flag1 or flag2:
                for root, dir, files in os.walk(f2_):
                    buggy_path = os.path.join(f2_, "neq")
                    files = glob.glob(buggy_path + '/*')
                    # for file in files:
                    #     print(file)


if __name__ == '__main__':
    main()