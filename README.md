# YoloV8_2Dpose

2d pose estimator using yolov8+hrnet.

## Installation

```
git clone ...
cd yolov8_2dpose
pip install -r requirements.txt
```

## Pretrained Model

Load [pose_hrnet_w32_256x192.pth](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) to `yolov8_2dpose/joints_detectors/hrnet/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth`

## Usage

```
python inference.py --video path/to/video
```

## Reference

[https://github.com/mikel-brostrom/yolov8_tracking](https://github.com/mikel-brostrom/yolov8_tracking)

[https://github.com/lxy5513/videopose](https://github.com/lxy5513/videopose)
