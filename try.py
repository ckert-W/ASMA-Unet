import argparse
from data import test_dataloader
parser=argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str,default='datas/dataa')
args=parser.parse_args()
dataloader = test_dataloader(args.input_dir, batch_size=1, num_workers=10)

for iter_idx, data in enumerate(dataloader):
    input_img, label_img, name = data