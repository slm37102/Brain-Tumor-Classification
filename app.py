import pandas as pd
from fastai.vision.all import *
import time

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def get_x(r):
    return r['filepath']

def get_y(r):
    return r['MGMT_value']

learn = load_learner('export.pkl')
start_time = time.time()
print(learn.predict('Image-13.png'))
print(time.time() - start_time)
