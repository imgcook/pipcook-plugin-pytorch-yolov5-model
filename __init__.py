import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import yaml
import logging
import requests
import torch
from pathlib import Path

from models.yolo import Model
from utils.torch_utils import select_device, intersect_dicts

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

def define(hyp, opt, device):
  logger.info(f'Hyperparameters {hyp}')
  log_dir = './evolve'

  os.makedirs(log_dir, exist_ok=True)
  with open(log_dir + '/hyp-define.yaml', 'w') as f:
    yaml.dump(hyp, f, sort_keys=False)
  with open(log_dir + '/opt-define.yaml', 'w') as f:
    yaml.dump(vars(opt), f, sort_keys=False)

  
  weights = opt.weights
  modelPath = os.path.join(BASE_PATH, opt.weights)
  modelDownloadUrl = os.path.join(MODEL_URL, opt.weights)
  if not os.path.exists(modelPath):
    download(modelDownloadUrl, BASE_PATH, opt.weights)

  ckpt = torch.load(modelPath, map_location=device)
  if hyp.get('anchors'):
      ckpt['model'].yaml['anchors'] = round(hyp['anchors'])
  model = Model(os.path.join(os.path.dirname(__file__), 'models', 'yolov5s.yaml'), ch=3, nc=opt.nc).to(device)
  exclude = ['anchor']
  state_dict = ckpt['model'].float().state_dict()  # to FP32
  state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
  model.load_state_dict(state_dict, strict=False)
  logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
  return model, ckpt

def main(data, args):
  opt = obj({})
  opt.hyp = os.path.join(os.path.dirname(__file__), 'config/hyp.scratch.yaml')
  opt.device = ''
  opt.total_batch_size = 16 if not hasattr(args, 'batch_size') else args.batch_size
  opt.batch_size = opt.total_batch_size
  opt.nc = len(vars(data.metadata.labelMap))
  opt.weights = 'yolov5s.pt'
  opt.cfg = 'yolov5s.yaml'

  with open(opt.hyp) as f:
      hyp = yaml.load(f, Loader=yaml.FullLoader)
  device = select_device(opt.device, batch_size=opt.batch_size)
  yolov5, ckpt = define(hyp, opt, device)
  sys.path.pop()
  sys.path.pop()
  class PipcookModel:
    model = yolov5
    config = {
      "ckpt": ckpt
    }

  return PipcookModel()
