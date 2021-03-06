# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

#import tensorflow as tf
#from nets.resnet_v1 import resnetv1

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  '''
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  '''
  parser.add_argument('--json_path', dest='json_path')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  import json
  json_data = json.load(open(args.json_path, "r"))
  return json_data


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb

import os
if __name__ == '__main__':
  args = parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
  #print('Called with args:')
  #print(args)

  if args.get('cfg_file') is not None:
    cfg_from_file(args['cfg_file'])
  if args.get('set_cfgs') is not None:
    cfg_from_list(args['set_cfgs'])

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  output_dir = args['working_dir']
  cfg.TRAIN.CACHE_PATH = output_dir
  # train set
  imdb_list = []
  roidb_list = []
  for task in args['task_list']:
    imdb, roidb = combined_roidb(task['imdb'])
    print('{:d} roidb entries'.format(len(roidb)))
    task['num_classes'] = imdb.num_classes
    imdb_list.append(imdb)
    roidb_list.append(roidb)

  # output directory where the models are saved
  #output_dir = get_output_dir(imdb_list[0], None)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  #tb_dir = get_output_tb_dir(imdb_list[0], None)
  tb_dir = output_dir
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  valroidb_list = []
  ''' 
  for task in args['task_list']:
    print("load " + task['imdbval'])
    _, valroidb = combined_roidb(task['imdbval'])
    valroidb_list.append(valroidb)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  '''
  # load network
  #from nets.mult_vgg16 import vgg16
  #if args['net'] == 'vgg16':
  #  net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  #else:
  #  raise NotImplementedError
    
  train_net(args['net'], args['task_list'], roidb_list, valroidb_list, output_dir, tb_dir,
            pretrained_model=args['weight'],
            max_iters=args['iters'])
