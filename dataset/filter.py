import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
path = './dataset/building/train/'
fi_names = os.listdir(path)
f_path = pd_file['file_name'].tolist()

class_names = pd_file['classes'].astype(int).tolist()
train_files = [path+"/wikiart/"+f_path[i] for i in range(len(f_path))]

for i in tqdm(range(0, len(train_files))):
    img = Image.open(train_files[i])
    img.save(f"{path}/test/{f_path[i].split('/')[-1]}", quality=100)
