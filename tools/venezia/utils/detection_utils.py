from collections import deque
import cv2
from threading import Thread
import numpy as np
import time
import torch
import torchvision
import os
import logging
import requests

#code partly adapted from https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

def on_segment(p, q, r):
# check if point r is on segment pq
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
# return 0/1/-1 for colinear/clockwise/counterclockwise
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
# check if seg1 and seg2 intersect
    p1, q1 = seg1
    p2, q2 = seg2
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:               return o2
    if o1 == 0 and on_segment(p1, q1, p2) : return 0  # avoid double counting if one point is on the barrier
    if o2 == 0 and on_segment(p1, q1, q2) : return -o1
    if o3 == 0 and on_segment(p2, q2, p1) : return o2
    if o4 == 0 and on_segment(p2, q2, q1) : return o2
    return 0

def bbox_rel(image_width, image_height,  *xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    # Performs Non-Maximum Suppression (NMS) on inference results
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print('nms error', x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def select_device(logger, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logging.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                        (s, i, x[i].name, x[i].total_memory / c))
    else:
        logging.info('Using CPU')
    logging.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources, img_size=640, buff_len=100, logger=None, masks=None):
        self.mode = 'images'
        self.img_size = img_size
        self.logger = logger

        n = len(sources)
        self.buff_len = buff_len
        self.imgs = [deque(maxlen=buff_len) for x in range(n)] # ring buffer
        self.status = [False] * n
        self.sources = sources

        self.mask_on = [len(m)>2 for m in masks]
        self.masks = masks
        self.mask_img = [None] * n

        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            self.logger.info('%g/%g: %s... ' % (i + 1, n, s))
            thread = Thread(target=self.update, args=([i]), daemon=True)
            thread.start()

    def update(self, index):
        c_status = False # camera open
        s = self.sources[index]
        ip = s.split('@')[1].split('/')[0]
        cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
        if cap is None or not cap.isOpened():
          self.logger.info(f'failed to open {s}')
        else:
          fps = cap.get(cv2.CAP_PROP_FPS) #%100
          w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          self.logger.info(f'DT - {s} success ({w}x{h} at {fps:.2f} FPS)')
          c_status = True
        # Read next stream frame in a daemon thread
        while True:
          r = False
          try:
            r, img = cap.read()
          except cv2.error as e:
            self.logger.info(f'DT - {ip} camera error: {e}')
          if r==True:
            if self.mask_on[index]: # masking
              if self.mask_img[index] is None: # must initialize mask image the first time
                self.mask_img[index] = np.zeros(img.shape[:2], dtype="uint8")
                cv2.fillPoly(self.mask_img[index], pts = np.array([self.masks[index]]), color=(255,255,255))
              img = cv2.bitwise_and(img, img, mask=self.mask_img[index])
            self.imgs[index].append(img)
            self.status[index] = True # new frames to process
            time.sleep(0.001)  # wait time
          else:
            self.status[index] = False
            # use this try block if requests can get the camera status
            """
            try:
              pingtry = requests.get(f'http://{ip}',timeout=10)
              if pingtry.status_code == requests.codes.ok:
                if c_status==False:
                  cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
                  if cap is None or not cap.isOpened():
                    self.logger.info(f'DT - {ip} reconnect failed')
                    c_status = False
                  else:
                    self.logger.info(f'DT - {ip} reconnect success')
                    c_status = True
                  continue
              else:
                self.logger.info(f'DT - {ip} camera offline')
            """
            # use this try block instead if cam status is only accessible by CV
            try:
                if c_status==False: # allow 1 missed frame
                  cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
                  if cap is None or not cap.isOpened():
                    self.logger.info(f'DT - {ip} reconnect failed')
                    time.sleep(60)  # wait time before retrying
                  else:
                    self.logger.info(f'DT - {ip} reconnect success')
                    c_status = True
                  continue
            except cv2.error as e:
              self.logger.ingo(f'DT - {ip} cv error: {e}')
            except Exception as e:
              self.logger.info(f'DT - {ip} runtime error : {e}')
            c_status = False



    def grab(self):
        imgl, img0l = [], []
        for i,s in enumerate(self.sources):
          if self.status[i]:
            img0 = self.imgs[i].copy()
            img0 = np.asarray(img0,dtype= np.uint8)

            # Letterbox
            img = [letterbox(x, new_shape=self.img_size)[0] for x in img0]
            # Stack
            img = np.stack(img, 0)
            # Convert
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)
            imgl.append(img)
            img0l.append(img0)
        return  imgl, img0l
