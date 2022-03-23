import os
import sys


def truncate(filename, output_file):
    object_grasp = {}


    list
    with open(filename, "r") as file:

        with open(output_file, "w") as output:

            for line in file:
                temp = line.rstrip().split("-", 2)[:2]
                if temp not in object_grasp:
                    object_grasp[temp] = 1


folder_name = sys.argv[1]
output_file = sys.argv[2]

file_name = "train_split.txt"
file_name_test = "test_split.txt"

if os.path.exists(folder_name):
    train_file = os.path.join(folder_name, file_name)
    test_file = os.path.join(folder_name, file_name_test) 
else:
    exit()

print(train_file)
truncate(train_file)