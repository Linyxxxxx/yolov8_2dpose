import os

from tools.api import inference
from argparse import ArgumentParser

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--video", help="input video file name")
    args = parser.parse_args()
    inference(args.video, device='0')