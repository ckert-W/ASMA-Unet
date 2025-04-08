import argparse
from PIL import Image as Image
import cv2
import torch
import os
from models.MIMOUNet import build_net
from torch.backends import cudnn
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

def main(args):
    # CUDNN
    cudnn.benchmark = True
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    model.cuda()
    image=Image.open(args.input_dir)
    transform = transforms.ToTensor()
    image=transform(image)
    image = image.to(torch.device('cuda'))

    image=torch.unsqueeze(image,dim=0)

    with torch.no_grad():
        outputs = model(image)[2]
        outputs_clip=torch.clamp(outputs,0,1)
        outputs_clip+= 0.5 / 255
        outputs = F.to_pil_image(outputs_clip.squeeze(0).cpu(),'RGB')
        outputs.save(args.result_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus'], type=str)
    parser.add_argument('--input_dir',type=str,default='datas/dataa/test/blur/0240406123601.jpg')
    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')

    print(args)
    main(args)