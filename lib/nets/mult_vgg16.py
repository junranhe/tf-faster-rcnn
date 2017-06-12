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
    self._layers = {}

    self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])

    outputs = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=is_reuse):
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
          task['im_info_ph'] = tf.placeholder(tf.float32, shape=[self._batch_size, 3]) 
          task['gt_boxes_ph'] = tf.placeholder(tf.float32, shape=[None, 5])
          task['image_ph'] = self._image
          out = task['net'].create_architecture(sess, mode, task['num_classes'], self._image, task['im_info_ph'], task['gt_boxes_ph'],\
                                          self._layers['head'], tag, anchor_scales, anchor_ratios)
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
      
