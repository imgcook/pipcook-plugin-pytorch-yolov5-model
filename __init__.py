import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import cv2
import yaml
import logging
import requests
import torch
import numpy as np
import json
from pathlib import Path

from models.yolo import Model
from models.experimental import attempt_load
from utils.torch_utils import select_device, intersect_dicts
from utils.datasets import letterbox
from utils.general import scale_coords, non_max_suppression

logger = logging.getLogger(__name__)

MODEL_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/yolov5s'
BASE_PATH = os.path.join(str(Path.home()), '.pipcook', 'yolov5')

class obj(object):
  def __init__(self, d):
    for a, b in d.items():
      if isinstance(b, (list, tuple)):
          setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
      else:
          setattr(self, a, obj(b) if isinstance(b, dict) else b)

def download(url, folderpath, filepath):
  r = requests.get(url, allow_redirects=True)
  if not os.path.exists(folderpath):
    os.makedirs(folderpath)
  with open(os.path.join(folderpath, filepath), 'wb+') as f:
    f.write(r.content)

def define(hyp, opt, device, recoverPath):
  logger.info(f'Hyperparameters {hyp}')
  log_dir = './evolve'

  os.makedirs(log_dir, exist_ok=True)
  with open(log_dir + '/hyp-define.yaml', 'w') as f:
    yaml.dump(hyp, f, sort_keys=False)
  with open(log_dir + '/opt-define.yaml', 'w') as f:
    yaml.dump(vars(opt), f, sort_keys=False)

  
  weights = opt.weights
  if recoverPath:
    modelPath = os.path.join(recoverPath, 'weights', 'best.pt')
  else:
    modelPath = os.path.join(BASE_PATH, opt.weights)
    modelDownloadUrl = os.path.join(MODEL_URL, opt.weights)
    if not os.path.exists(modelPath):
      download(modelDownloadUrl, BASE_PATH, opt.weights)

  ckpt = torch.load(modelPath, map_location=device)
  if hyp.get('anchors'):
      ckpt['model'].yaml['anchors'] = round(hyp['anchors'])
  predict_model = attempt_load(modelPath, map_location=device)
  model = Model(os.path.join(os.path.dirname(__file__), 'models', 'yolov5s.yaml'), ch=3, nc=opt.nc).to(device)
  exclude = ['anchor']
  state_dict = ckpt['model'].float().state_dict()  # to FP32
  state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
  model.load_state_dict(state_dict, strict=False)
  logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
  return model, predict_model, ckpt

def main(data, args):
  opt = obj({})
  recoverPath = None if not hasattr(args, 'recoverPath') else args.recoverPath
  opt.hyp = os.path.join(os.path.dirname(__file__), 'config/hyp.scratch.yaml')
  opt.device = ''
  opt.total_batch_size = 16 if not hasattr(args, 'batch_size') else args.batch_size
  opt.batch_size = opt.total_batch_size
  if recoverPath:
    with open(os.path.join(recoverPath, '..', 'metadata.json')) as f:
      log = json.load(f)
      labelMap = json.loads(log['output']['dataset'])
      opt.nc = len(labelMap)
  else:
    opt.nc = len(vars(data.metadata.labelMap))
  opt.weights = 'yolov5s.pt'
  opt.cfg = 'yolov5s.yaml'
  opt.img_size = [640, 640] if not hasattr(args, 'imgSize') else args.imgSize

  with open(opt.hyp) as f:
      hyp = yaml.load(f, Loader=yaml.FullLoader)
  device = select_device(opt.device, batch_size=opt.batch_size)
  yolov5, predict_model, ckpt = define(hyp, opt, device, recoverPath)
  half = device.type != 'cpu'
  class PipcookModel:
    model = yolov5
    p_model = predict_model
    config = {
      "ckpt": ckpt,
      "img_size": opt.img_size
    }
    def predict(self, inputData):
      img_origin = cv2.imread(inputData.data)
      img = letterbox(img_origin, new_shape=opt.img_size)[0]
      # Convert
      img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      img = np.ascontiguousarray(img)
      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      # Inference
      pred = self.p_model(img)[0]
      pred = non_max_suppression(pred, 0.25, 0.45)
      
      # Parse Inference
      boxes = []
      classes = []
      scores = []
      for i, det in enumerate(pred):  # detections per image
        # Write results
        for *xyxy, conf, cls in reversed(det):
          boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
          classes.append(int(cls))
          scores.append(float(conf))
      output = {
        'boxes': boxes,
        'classes': classes,
        'scores': scores
      }
      return output

  sys.path.pop()
  sys.path.pop()
  return PipcookModel()
