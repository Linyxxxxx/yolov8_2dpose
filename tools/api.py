import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / '..' / 'joints_detectors' / 'yolov8_tracking'
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from joints_detectors.yolov8_tracking.yolov8.ultralytics.nn.autobackend import AutoBackend
from joints_detectors.yolov8_tracking.yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from joints_detectors.yolov8_tracking.yolov8.ultralytics.yolo.utils.checks import check_imgsz
from joints_detectors.yolov8_tracking.yolov8.ultralytics.yolo.utils.files import increment_path
from joints_detectors.yolov8_tracking.yolov8.ultralytics.yolo.utils.torch_utils import select_device
from joints_detectors.yolov8_tracking.yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes

from joints_detectors.yolov8_tracking.trackers.multi_tracker_zoo import create_tracker

from joints_detectors.hrnet.pose_estimation.video import get_pose_model, get_pose
from tools.utils import draw_2Dimg, resize_img


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov8n.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=[640, 640],  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        save_path='out.mp4',
):

    source = str(source)

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    pose_model = get_pose_model()

    # Dataloader
    bs = 1
    dataset = LoadImages(
        source,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None),
        vid_stride=vid_stride
    )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in tqdm(enumerate(dataset)):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1

            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path

            curr_frames[i] = im0
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                bboxes = []
                # draw boxes for visualization
                if len(outputs[i]) > 0:                   
                    for _, (output) in enumerate(outputs[i]):
                        
                        # bbox = output[0:4]
                        # id = output[4]
                        # cls = output[5]
                        # conf = output[6]
                        bboxes.append(np.array([output[0], output[1], output[2], output[3]], dtype=np.float32))

            resized_im, W, H, factor = resize_img(im0)
            bboxes = [bbox * factor for bbox in bboxes]
            
            joint_2d = get_pose(pose_model, resized_im, bboxes)
            im0 = draw_2Dimg(resized_im, joint_2d)

            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
            vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]


def inference(image_path, device):
    tracking_method = 'bytetrack'
    tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
    run(
        source=image_path,
        yolo_weights=WEIGHTS / 'yolov8n.pt',
        tracking_method=tracking_method,
        tracking_config=tracking_config,
        conf_thres=0.5,
        iou_thres=0.5,
        device=device,
        classes=0,
    )