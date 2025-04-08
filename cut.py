#to make sure the sizes of images are multiple of 4.
#all massed up,its like a herb now...
import argparse
import numpy as np
import cv2
import os

parser=argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str,default='data_in/test/')
parser.add_argument('--out_dir',type=str,default='datas/dataa')
args=parser.parse_args()

def update(input_img_path):
    image=cv2.imread(input_img_path)
    #cropped = image[0:600,0:960]
    #cv2.imwrite(out_img_path,cropped)
image_filenames=[(os.path.join(args.input_dir , x))
                 for x in os.listdir(args.input_dir)]
for path in image_filenames:
    print(path)
    image=cv2.imread(path)

'''for path in image_filenames:
    print(path)
    update(path[0],path[1])'''