import numpy as np
import csv
import time
import os
import sys
import os.path
import math as m
import shutil
import cv2

base_path = './models/result8/'
for i in range(7518):
    file_path = base_path + str(i).zfill(6) + '.txt'
    if not os.path.isfile(file_path):
        print(file_path)
        with open(file_path, 'w') as det_file:
            det_file.write('')
    cv2.waitKey(1)