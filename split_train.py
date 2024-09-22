import random
import numpy as np
import os

if __name__=="__main__":
    folder_dirs='dataset/NILM/train/'
    for foldername in os.listdir(folder_dirs):
        file_dirs=folder_dirs+foldername
        file_list=[]
        for filename in os.listdir(file_dirs):
            file_list.append(filename.split('.')[0])
        
        random.shuffle(file_list)
        rate=0.2
        nums=rate*len(file_list)
        test_list=file_list[:nums]

        is_dir='dataset/NILM/test/'+foldername
        if not os.path.isdir(is_dir):
            os.makedirs(is_dir)

        for files in test_list:
            os.replace(file_dirs+files,is_dir+files)