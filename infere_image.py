from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import pdb
import matplotlib.pyplot as plt
from PIL import Image

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained were scaled by 5.4 to ease the training
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
TRANS_SCALE_FACTOR = 5.4


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate():
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    # assert os.path.isdir(opt.load_weights_folder), \
    #     "Cannot find a folder at {}".format(opt.load_weights_folder)

    # print("-> Loading weights from {}".format(opt.load_weights_folder))

    # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    # encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    # decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    # encoder_dict = torch.load(encoder_path)

    # dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
    #                                     encoder_dict['height'], encoder_dict['width'],
    #                                     [0], 4, use_imu=False, is_train=False)
    # dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(50, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    # depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = None

    # print("-> Computing predictions with size {}x{}".format(
    #     encoder_dict['width'], encoder_dict['height']))

    image = Image.open('assets/test.png').convert('RGB')
    w, h = image.size
    cw = int(w / 2)
    ch = int(h / 2)
    wr = w - w % 32
    hr = h - h % 32
    image = image.crop((cw - int(wr / 2), ch - int(hr / 2), cw + int(wr / 2), ch + int(hr / 2)))

    with torch.no_grad():
        input_color = torch.tensor(np.asarray(image)).movedim(-1, 0).unsqueeze(0).cuda()

        # if opt.post_process:
        #     # Post-processed results require each image to have two forward passes
        #     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

        output = depth_decoder(encoder(input_color))

        pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
        pred_disp = pred_disp.cpu()[:, 0].numpy()

        # N = pred_disp.shape[0] // 2
        # pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

    # pred_disps = np.concatenate(pred_disps)

    pred_disp = pred_disp[0]
    pred_depth = 1 / pred_disp

    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    plt.imshow(pred_depth)
    plt.show()


if __name__ == "__main__":
    # options = MonodepthOptions()
    evaluate('weights/')
