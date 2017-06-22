# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np
from tensorflow.python.ops import variable_scope
#from nets.mult_network import Network
import nets.mult_network as mult_network
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from model.bbox_transform import clip_boxes, bbox_transform_inv

from utils.shape_util import print_shape

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def build_fc_net(rpn_pooled_net,is_training):
  '''
    add pooled net
    Args:
      rpn_pooled_ne
    Return:
      net out to connect classifier 
  '''
  print_shape(rpn_pooled_net)
  pool5_flat = slim.flatten(rpn_pooled_net, scope='flatten')
  print_shape(pool5_flat)
  
  fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
  if is_training:
    fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
  print_shape(fc6)

  fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
  if is_training:
    fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
  print_shape(fc7)

  return fc7


class vgg16(object):
  def __init__(self, batch_size=1):
    #Network.__init__(self, batch_size=batch_size)
    self._batch_size = batch_size
    self._variables_to_fix = {}

  def get_task_net(self,task_id):
    return self._tasks[task_id]['net']

  def get_task(self, task_id):
    return self._tasks[task_id]

  def create_mult_architecture(self, sess, mode,\
                           task_list,\
                           tag=None,
                           anchor_scales=(8, 16, 32),\
                           anchor_ratios=(0.5, 1, 2),
                           is_reuse=False):
    self._tasks = [{'net':mult_network.Network(batch_size=self._batch_size),"num_classes":task_num_classes,\
                    "predictions":{},'anchor_targets':{}, 'proposal_targets':{}, 'im_info_ph':None, 'gt_boxes_ph':None} \
                   for task_num_classes in task_list]
    #task['im_info_ph'] = tf.placeholder(tf.float32, shape=[self._batch_size, 3]) 
    #task['gt_boxes_ph'] = tf.placeholder(tf.float32, shape=[None, 5])
    noreuse_list = []
    for task_id , task in enumerate(self._tasks):
      task['im_info_ph'] = tf.placeholder(tf.float32, shape=[self._batch_size, 3]) 
      task['gt_boxes_ph'] = tf.placeholder(tf.float32, shape=[None, 5])
      noreuse_list.append(task['im_info_ph'])
      noreuse_list.append(task['gt_boxes_ph'])
    self._layers = {}

    self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
    noreuse_list.append(self._image)
    outputs = []
    with tf.variable_scope(tf.get_variable_scope(), noreuse_list, reuse=is_reuse):
      with slim.arg_scope(mult_network.faster_rcnn_arg_scope()):
        return self.build_network(sess, mode,tag, anchor_scales, anchor_ratios, reuse=is_reuse, is_training=(mode == 'TRAIN'))

  def build_network(self, sess,mode,tag, anchor_scales, anchor_ratios, reuse=False, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                        trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')
      #self._act_summaries.append(net)
      self._layers['head'] = net
      # build the anchors for the image
# rcnn
      outputs = []
      for task_id, task in enumerate(self._tasks): 
        with tf.variable_scope(('branch_%d' % task_id), reuse=reuse):
          
          task['image_ph'] = self._image
          out = task['net'].create_architecture(sess, mode, task['num_classes'], self._image, task['im_info_ph'], task['gt_boxes_ph'],\
                                          self._layers['head'], build_fc_net, tag, anchor_scales, anchor_ratios)
          task['losses'] = out
          outputs.append(out)
      return outputs
      #return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []
    branch_weight_names = []
    for task_id, task in enumerate(self._tasks):
       branch_weight_names.append('vgg_16/branch_%d/fc6/weights:0' % task_id)
       branch_weight_names.append('vgg_16/branch_%d/fc7/weights:0' % task_id)

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name in branch_weight_names:
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'vgg_16/conv1/conv1_1/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv, 
                                      "vgg_16/fc7/weights": fc7_conv,
                                      "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
        for task_id , _ in enumerate(self._tasks):
          sess.run(tf.assign(self._variables_to_fix['vgg_16/branch_%d/fc6/weights:0' % task_id], tf.reshape(fc6_conv, 
                            self._variables_to_fix['vgg_16/branch_%d/fc6/weights:0' % task_id].get_shape())))
          sess.run(tf.assign(self._variables_to_fix['vgg_16/branch_%d/fc7/weights:0' % task_id], tf.reshape(fc7_conv, 
                            self._variables_to_fix['vgg_16/branch_%d/fc7/weights:0' % task_id].get_shape())))
  def test_image(self, sess, image, im_info, task_ids):
    feed_dict = {self._image: image}
    preds = []
    for task_id in task_ids:
      task = self._tasks[task_id]
      _, class_prob, box_pred, roi = task['net'].get_predictions()
      preds += [class_prob, box_pred, roi]
      
      feed_dict[task['im_info_ph']] = im_info
    outputs =  sess.run(preds, feed_dict=feed_dict)
    results = []
    for i in range(len(task_ids)):
      scores = outputs[3*i]
      bbox_pred =outputs[3*i + 1]
      rois = outputs[3*i + 2]
      results.append((scores, bbox_pred, rois))
    return results
