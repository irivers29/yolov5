import argparse
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadNumpyImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def run_from_python(weights=ROOT / 'MT_weights.pt',  source=ROOT / 'test/now', imgsz=640, conf_thres=0.25,
                    iou_thres=0.45, max_det=1000, device='', view_img=False, save_txt=True, save_conf=False, 
                    save_crop=True, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, 
                    update=False, project=ROOT / 'runs/detect', name='exp', exist_ok=False, line_thickness=3, hide_labels=True,
                    hide_conf=False, half=False, dnn=False):

    opt = parse_opt()
    crops = main(weights,  source, imgsz, conf_thres,
                    iou_thres, max_det, device, view_img, save_txt, save_conf, 
                    save_crop, nosave, classes, agnostic_nms, augment, visualize, 
                    update, project, name, exist_ok, line_thickness, hide_labels,
                    hide_conf, half, dnn)
    return crops

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'test/now', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    #parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(FILE.stem, opt)
    #print(opt)
    return opt


def main(weights=ROOT / 'MT_weights.pt',  source=ROOT / 'test/now', imgsz=640, conf_thres=0.25,
                    iou_thres=0.45, max_det=1000, device='', view_img=False, save_txt=True, save_conf=False, 
                    save_crop=True, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, 
                    update=False, project=ROOT / 'runs/detect', name='exp', exist_ok=False, line_thickness=3, hide_labels=True,
                    hide_conf=False, half=False, dnn=False, return_from_main = True):

    check_requirements(exclude=('tensorboard', 'thop'))
    if return_from_main:
        crops = run(weights,  source, imgsz, conf_thres,
                        iou_thres, max_det, device, view_img, save_txt, save_conf, 
                        save_crop, nosave, classes, agnostic_nms, augment, visualize, 
                        update, project, name, exist_ok, line_thickness, hide_labels,
                        hide_conf, half, dnn)
        return crops
    else:
        run(weights,  source, imgsz, conf_thres,
                iou_thres, max_det, device, view_img, save_txt, save_conf, 
                save_crop, nosave, classes, agnostic_nms, augment, visualize, 
                update, project, name, exist_ok, line_thickness, hide_labels,
                hide_conf, half, dnn)

@torch.no_grad()
def run(weights=ROOT / 'MT_weights.pt',  # model.pt path(s)
        source=ROOT / 'test/now',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    if type(source) == np.ndarray:
        numpy_input = True
    
    else:
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    imgsz = check_img_size(imgsz, s=stride)
    # Dataloader
    dataset = LoadNumpyImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)


    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for im in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            seen += 1
            
            im0, frame = source.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size

                # det[:,:4] is the bounding box 
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                crops = {}
                counter = 0
                for *xyxy, conf, cls in reversed(det):
                    #print(*xyxy)
                    crop_info = [xyxy]
                    crop = save_one_box(xyxy, imc, save=False, BGR=True)
                    crop_info.append(crop)
                    crops[counter] = crop_info

                    counter += 1
                    
                return crops
                # Write results
                #for *xyxy, conf, cls in reversed(det):
                #    if save_txt:  # Write to file
                #        print("xyxy", xyxy)
                #        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #        print("xywh", xywh)
                #        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #        with open(txt_path + '.txt', 'a') as f:
                #            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                #    if save_img or save_crop or view_img:  # Add bbox to image
                #        c = int(cls)  # integer class
                #        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                #        annotator.box_label(xyxy, label, color=colors(c, True))
                #        if save_crop:
                #            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

