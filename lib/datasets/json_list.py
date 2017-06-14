# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from model.config import cfg
import json

class json_list(imdb):
    def __init__(self, json_file):
        assert os.path.exists(json_file), \
                'file does not exist: {}'.format(json_file)

        imdb.__init__(self, os.path.splitext(os.path.basename(json_file))[0])

        with open(json_file) as f:
            json_config = json.loads(f.read())
            self._classes = [str(c) for c in range(json_config['num_labels'] + 1)]
            self.list_file = json_config['fileListJSON']
            self.json_file = json_file

        #cfg.DATA_DIR = os.path.dirname(self.json_file)
        self._data_dir = os.path.dirname(self.json_file)
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self.list_file), \
                'file does not exist: {}'.format(self.list_file)

        self._image_paths = []
        self._json_lines = []
        with open(self.list_file) as f:
            for line in f.readlines()[1:]:
                items = line.strip().split('\t')
                if len(items) == 2:
                    self._image_paths.append(items[0])
                    self._json_lines.append(items[1])
            if len(self._image_paths) % 2 == 1:
                self._image_paths = self._image_paths[:-1]
                self._json_lines = self._json_lines[:-1]
            self._image_index = range(len(self._image_paths))

    def image_path_at(self, i):
        index = self._image_index[i] % len(self._image_paths)
        return self._image_paths[index]
    @property
    def cache_path(self):
      cache_path = osp.abspath(osp.join(self._data_dir, 'cache'))
      if not os.path.exists(cache_path):
        os.makedirs(cache_path)
      return cache_path


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_json_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_json_annotation(self, index):
        """
        Load image and bounding boxes info from json lines.
        """
        objs = json.loads(self._json_lines[index])
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj['xmin'])
            y1 = float(obj['ymin'])
            x2 = float(obj['xmax'])
            y2 = float(obj['ymax'])
            cls = int(obj['name']) + 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


