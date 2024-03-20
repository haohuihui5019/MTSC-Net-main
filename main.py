import argparse
import torch
import os


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--label-info', default=None, type=str, help='the path to the label information')
    parser.add_argument('--ratio', default=0.7, type=float,
                        help='The mask ratio of the Augmentation technique')
    parser.add_argument('--size', default=64, type=int,
                        help='The size of the mask square')
    parser.add_argument('--downsample-ratio', default=8, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--data-dir', default='', help='the directory of the data')
    parser.add_argument('--save-dir', default='')
    parser.add_argument('--resume', default="")
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--labeled', default=4, type=int, )
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='the learning rate')
    parser.add_argument('--device', default='', help="assign device")
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, help='the number of starting epoch')
    parser.add_argument('--epochs', default=1, type=int,
                        help='the maximum number of training epoch')
    parser.add_argument('--start-val', default=20, type=int,
                        help='the starting epoch for validation')
    parser.add_argument('--val-epoch', default=20, type=int,
                        help='the number of epoch between validation')
    parser.add_argument('--weight-ramup', default=20, type=int)

    args = parser.parse_args()
    return args


