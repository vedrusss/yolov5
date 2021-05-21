__author__="Alexey Antonenko, a.antonenko@aiby.com"
import argparse
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized

def get_detector(weights, device, imgsz, verbose=True):
    # Initialize
    torch.no_grad()
    if verbose: set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    if device.type != 'cpu':  # Warm up the device
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model, half, stride, names, device

def load_image(cv_bgr_image, img_size, stride):
    # Padded resize
    img = letterbox(cv_bgr_image, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, cv_bgr_image.shape

def detect(model, names, augment, conf_thres, iou_thres, classes, agnostic_nms, img, src_img_shape, half, device, verbose=False):
    t0 = time.time()
    #  put to device
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    #  preprocess
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()
    # Print time (inference + NMS)
    if verbose:
        print(f'Inference + NMS done in {t2 - t1:.3f} secs')

    detections = [] 
    # Process detections
    det = pred[0]  #  preditions are returned for a batch of images
    if len(det):
        # Rescale boxes from img_size to im0s size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], src_img_shape).round()
        detections = [{'cls':names[int(cls)], 'score':float(conf), 'box':[int(el) for el in xyxy]} for *xyxy, conf, cls in reversed(det)]
    return detections

def main(opt):
    imgsz = opt.img_size
    model, half, stride, names, device = get_detector(opt.weights, opt.device, imgsz)    
    for path in opt.source:
        #  load image
        cv_bgr_image = cv2.imread(path)  # BGR
        img, src_img_shape = load_image(cv_bgr_image, imgsz, stride)
        detections = detect(model, names, opt.augment, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, img, src_img_shape, half, device)
        print(f"Results for {path}:")
        print(detections)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', nargs='+', type=str, required=True, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    main(opt)
