'''
    this script will use our.pkl to inference all target under given folder.
    output to ./result/
'''

import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
# from imageio import mimsave
import torch.nn.functional as F    
import config as cfg
from Trainer import Model

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', default='/kaggle/input/vimeo90k-dataset/topic3_release/private_test_set', type=str)
parser.add_argument('--scaleup', default=1, type=int) # using CV2 to upscaling
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Start Generating=========================')

# result folder
res_path = '/kaggle/working/result'

total=0
for first_level in os.listdir(args.path):
    first_level_path = os.path.join(args.path, first_level)
    if os.path.isdir(first_level_path):
        r1 = os.path.join(res_path, first_level)

        # 遍歷給定目錄的第二層
        for second_level in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level)
            if os.path.isdir(second_level_path):

                r2 = os.path.join(res_path, second_level)
                os.makedirs(r2, exist_ok=True)

                # 輸出目錄結構
                print(f"{first_level}/{second_level}")
                total += 1
                I0 = cv2.imread(os.path.join(second_level_path, 'im3.jpg'))
                I2 = cv2.imread(os.path.join(second_level_path, 'im5.jpg'))

                if args.scaleup:
                    I0 = cv2.resize(I0, (448, 256), interpolation=cv2.INTER_LINEAR)
                    I2 = cv2.resize(I2, (448, 256), interpolation=cv2.INTER_LINEAR)
                
                assert I0.shape[:2] == (448, 256), "ERROR: input should be 448x256"

                I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
                I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

                padder = InputPadder(I0_.shape, divisor=32)
                I0_, I2_ = padder.pad(I0_, I2_)

                I1 = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

                cv2.imwrite(os.path.join(r2, 'im4.png'), I1)

print(f'=========================Done=========================')
