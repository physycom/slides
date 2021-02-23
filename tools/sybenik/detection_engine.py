import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn


try:
  sys.path.append(os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik'))
  from models.deepsort import DeepSort
  from utils.detection_utils import LoadStreams
  from utils.detection_utils import select_device, scale_coords, non_max_suppression, bbox_rel, intersects
except Exception as e:
  raise Exception(f'detector : lib init failed {e}')

class detector:

  def __init__(self, configfile):
    with open(configfile) as cin:
      config = json.load(cin)


    self.wdir = config['workdir']
    self.datadir = f'{self.wdir}/detection_data'
    if not os.path.exists(self.datadir): os.mkdir(self.datadir)
    logfile = f'{self.wdir}/detection-engine.log'
    self.clock_dt = config['clock_dt']
    self.dump_dt = config['dump_dt']
    self.pending_detection = True

    logging.basicConfig(
      filename=logfile,
      filemode='w',
      level=logging.DEBUG,
      format='%(asctime)s [%(levelname)s] %(message)s',
      datefmt='%y-%m-%d %H:%M:%S%z'
    )
    self.config = config

    camdatafile = os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik', 'data', 'cam_data.json')
    with open(camdatafile) as cin:
      self.camcfg = json.load(cin)
    self.camdata = [c for c in self.camcfg if c['cam_cnt']]
    # Initialize model
    self.device = select_device(config['yolo']['device']) # cuda device, i.e. 0 or 0,1,2,3 or cpu
    self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
    self.model = torch.load(config['yolo']['weights'], map_location=self.device)['model'].float()  # load to FP32
    self.model.to(self.device).eval()
    if self.half:
      cudnn.benchmark = True  # set True to speed up constant image size inference
      self.model.half()  # to FP16

    # Initialize streams
    self.view_img = False
    self.imgsz = config['yolo']['inference_size']
    self.sources = [f"rtsp://{config['user']}:{config['pass']}@{c['ip']}/live" for c in self.camdata]
    self.ns = len(self.sources)
    self.dataset = LoadStreams(self.sources, img_size=self.imgsz, buff_len = config['frame_buffer'])

    # Initialize deepsort
    self.deepsort =  [DeepSort(config['deepsort']['REID_CKPT'],
                       max_dist=config['deepsort']['MAX_DIST'],
                       min_confidence=config['deepsort']['MIN_CONFIDENCE'],
                       nms_max_overlap=config['deepsort']['NMS_MAX_OVERLAP'],
                       max_iou_distance=config['deepsort']['MAX_IOU_DISTANCE'],
                       max_age=config['deepsort']['MAX_AGE'],
                       n_init=config['deepsort']['N_INIT'],
                       nn_budget=config['deepsort']['NN_BUDGET'],
                       use_cuda=True) for x in range(self.ns)]

    # Get class names
    self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    # Run inference
    img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
    _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    # Crossing variables
    # if barrier is X axis, IN is positive y and OUT is negative y
    self.barriers = [{} for x in range(self.ns)]
    
    for i,c in enumerate(self.camdata):
      maxw, maxh = self.dataset.dims[i]
      for btag,bcoords in c['barriers'].items():
        if not bcoords['enabled']:
          continue

        bc = bcoords['pixelcoords']
        if 0<=bc[0]<=maxw and 0<=bc[2]<=maxw and 0<=bc[1]<=maxh and 0<=bc[3]<=maxh:
          self.barriers[i][btag]=(((bc[0],bc[1]),(bc[2],bc[3])))
        else:
          logging.info('barrier is out of the image')
    
    # Dump helper
    self.dumper = {'cams':[[] for x in range(self.ns)], 'bars':[[] for x in range(self.ns)]}
    logging.info('init engine done')

  def do_task(self):
    try:
      tnow = datetime.now()
      ts = int(tnow.timestamp())
      if int(ts) % self.clock_dt == 0:
        if self.pending_detection:
          logging.info('performing detections')
          imgl, img0l = self.dataset.grab()

          for j in range(self.ns):
            cname = self.camdata[j]['name']
            logging.info(f'detection for : {cname}')
            img = imgl[j]    #img= 1 buffer of frames ready for net
            img0 = img0l[j]  #img0= 1 buffer of original frames
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3: # buffer has only 1 frame
              break

            # Inference
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred,
                     conf_thres = self.config['yolo']['nms_conf'],
                     iou_thres = self.config['yolo']['nms_iou'],
                     classes=self.config['yolo']['classes'], #0 is people only. change deepsort model if need to track other things
                     agnostic=self.config['yolo']['nms_agnostic'])
            # Detection output
            prev_outputs, cam_output, bar_output = [], [], {}
            # Check if there are enabled barriers
            if len(self.barriers[j])==0: # no tracking, only counting
              for i, det in enumerate(pred):  
                if det is not None and len(det)!=0:
                  cam_output.append(len(det)) 
                  
            else:
              logging.info(f'tracking for : {cname}')
              for btag in self.barriers[j]:
                bar_output[btag] = {'IN':0,'OUT':0}
  
              for i, det in enumerate(pred):  # detections per buffer
                if det is not None and len(det)!=0:
                  im0 = img0[i].copy()
                  cam_output.append(len(det)) # detections counts
                  
                  # Rescale boxes from img_size to im0 size
                  det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                  bbox_xywh = []
                  confs = []
  
                  # Adapt detections to deep sort input format
                  for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
  
                  xywhs = torch.Tensor(bbox_xywh)
                  confss = torch.Tensor(confs)
  
                  # Pass detections to deepsort
                  outputs = self.deepsort[j].update(xywhs, confss, im0)
  
                  # Calculate barrier crossing
                  if i==0:
                    prev_outputs = outputs
                  else:
                    if len(outputs)!=0 and len(prev_outputs)!=0:
                      prev_ids = prev_outputs[:,-1]
                      for output in outputs:
                        identity = output[-1]
                        if identity in prev_ids:
                          prev_output = prev_outputs[prev_outputs[:,-1]==identity][0]
                          c_p_x, c_p_y = (prev_output[0] + prev_output[2]) //2, (prev_output[1] + prev_output[3]) //2
                          c_x, c_y = (output[0] + output[2]) //2, (output[1] + output[3]) //2
  
                          for btag,bcoords in self.barriers[j].items():
                            crossing = intersects(bcoords,((c_p_x,c_p_y),(c_x,c_y)))
                            if crossing == 1:
                              bar_output[btag]['IN'] +=1
                            elif crossing == -1:
                              bar_output[btag]['OUT'] +=1
                    prev_outputs = outputs

            if len(cam_output) == 0:
              cam_output=[0]
            # Write results to file
            if self.config['json_dump'] == True:
              cam_dump = {
                'cam_name'  : cname,
                'timestamp' : ts,
                'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
                'counter'   : {'MEAN':np.mean(cam_output),
                               'MAX':int(np.max(cam_output)),
                               'MIN':int(np.min(cam_output))}
              }
              self.dumper['cams'][j].append(cam_dump)
              if len(bar_output)!=0:
                bar_dump = {
                  'cam_name'  : cname,
                  'timestamp' : ts,
                  'datetime'  : datetime.fromtimestamp(ts).strftime('%y%m%d %H%M%S'),
                  'counter'   : bar_output
                }
                self.dumper['bars'][j].append(bar_dump)
                
              if int(ts) % self.dump_dt == 0:
                logging.info(f'dumping for : {cname}')
                cdatafile = f'{self.datadir}/cam_{cname}_{ts}.json'
                with open(cdatafile, 'w') as cout:
                  json.dump(self.dumper['cams'][j], cout, indent=2)
                  self.dumper['cams'][j] = [] #reset
                if len(bar_output)!=0:
                  bdatafile = f'{self.datadir}/bar_{cname}_{ts}.json'
                  with open(bdatafile, 'w') as cout:
                    json.dump(self.dumper['bars'][j], cout, indent=2)
                    self.dumper['bars'][j] = []
          self.pending_detection = False
      else:
        if not self.pending_detection:
          self.pending_detection = True
    except Exception as e:
      print('runtime error: ',e)



def make_cam_map(filename):
  import pandas as pd
  import folium

  with open(filename) as jin:
    camdata = json.load(jin)

  df = pd.DataFrame.from_dict(camdata)
  df['lat'] = df.coords.apply(lambda x: x[0])
  df['lon'] = df.coords.apply(lambda x: x[1])
  df['fake_name'] = df.name
  df.loc[ df.name.str.startswith('Centro'), 'fake_name' ] = 'Centro 1 & Centro 2'
  df = df.drop(columns=['barriers', 'ip'])
  center = df[['lat', 'lon']].mean()
  print(df)
  m = folium.Map(location=center, control_scale=True, zoom_start=9)
  df.apply(lambda row: folium.CircleMarker(
    location=[row.lat, row.lon],
    radius=8,
    fill_color='purple',
    color='purple',
    fill_opacity=1.,
    popup=folium.Popup(f'camera {row["fake_name"]}', show=True, sticky=True),
  ).add_to(m), axis=1)
  s, w = df[['lat', 'lon']].min()
  n, e = df[['lat', 'lon']].max()
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(f'map_camera.html')


if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfile', help='engine config json', required=True)
  args = parser.parse_args()

  if 1:
    det = detector(configfile=args.cfile)
    while True:
      det.do_task()

  if 0:
    make_cam_map(os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik', 'data', 'cam_data.json'))
