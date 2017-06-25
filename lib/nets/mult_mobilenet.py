# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np
from tensorflow.python.ops import variable_scope
import nets.mult_network as mult_network
from model.config import cfg

from utils.shape_util import print_shape

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

Conv_Shape = namedtuple('Conv_Shape', ['beta_mean_vari'])
DepthSepConv_Shape = namedtuple('DepthSepConv_Shape', ['dep_w', 'dep_beta_mean_vari','point_w','point_beta_mean_vari'])

_CONV_SHAPES = [
    Conv_Shape(beta_mean_vari=[32]),
    DepthSepConv_Shape(dep_w=[3, 3, 32, 1],dep_beta_mean_vari=[32],point_w=[1, 1, 32, 64],point_beta_mean_vari=[64]),
    DepthSepConv_Shape(dep_w=[3, 3, 64, 1],dep_beta_mean_vari=[64],point_w=[1, 1, 64, 128],point_beta_mean_vari=[128]),
    DepthSepConv_Shape(dep_w=[3, 3, 128, 1],dep_beta_mean_vari=[128],point_w=[1, 1, 128, 128],point_beta_mean_vari=[128]),
    DepthSepConv_Shape(dep_w=[3, 3, 128, 1],dep_beta_mean_vari=[128],point_w=[1, 1, 128, 256],point_beta_mean_vari=[256]),
    DepthSepConv_Shape(dep_w=[3, 3, 256, 1],dep_beta_mean_vari=[256],point_w=[1, 1, 256, 256],point_beta_mean_vari=[256]),
    DepthSepConv_Shape(dep_w=[3, 3, 256, 1],dep_beta_mean_vari=[256],point_w=[1, 1, 256, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 512],point_beta_mean_vari=[512]),
    DepthSepConv_Shape(dep_w=[3, 3, 512, 1],dep_beta_mean_vari=[512],point_w=[1, 1, 512, 1024],point_beta_mean_vari=[1024]),   
    DepthSepConv_Shape(dep_w=[3, 3, 1024, 1],dep_beta_mean_vari=[1024],point_w=[1, 1, 1024, 1024],point_beta_mean_vari=[1024])
]

BRANCH_START_IDX = 12

def build_fc_net_without_conv(rpn_pooled_net,is_training):
  fc7 = tf.reduce_mean(rpn_pooled_net,axis=[1,2])
  return fc7


def build_fc_net(rpn_pooled_net,is_training,min_depth=8,depth_multiplier=1.0):
  print_shape(rpn_pooled_net)
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  net = rpn_pooled_net #init net with head, fix a big bug here

  with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
    for net_idx in range(BRANCH_START_IDX,len(_CONV_DEFS)):
      conv_def = _CONV_DEFS[net_idx]
      end_point_base = 'Conv2d_%d' % net_idx
      end_point = end_point_base + '_depthwise'
      net = slim.separable_conv2d(net, None, conv_def.kernel,
                                  depth_multiplier=1,
                                  stride=conv_def.stride,
                                  normalizer_fn=slim.batch_norm,
                                  scope=end_point,
                                  trainable=is_training)
      print_shape(net)

      end_point = end_point_base + '_pointwise'
      net = slim.conv2d(net, conv_def.depth, [1, 1],
                        stride=1,
                        normalizer_fn=slim.batch_norm,
                        scope=end_point,
                        trainable=is_training)
      print_shape(net)

    fc7 = tf.reduce_mean(net,axis=[1,2])
    print_shape(fc7)
    return fc7

    
def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None,
                      is_training=True):
  """Mobilenet v1.
  Constructs a Mobilenet v1 network from inputs to the given final endpoint.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
      'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
      'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
      'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
      'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      # The current_stride variable keeps track of the output stride of the
      # activations, i.e., the running product of convolution strides up to the
      # current network layer. This allows us to invoke atrous convolution
      # whenever applying the next convolution would result in the activations
      # having output stride larger than the target output_stride.
      current_stride = 1

      # The atrous convolution rate parameter.
      rate = 1

      net = inputs
      print_shape(net,'input shape ')
      for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i
        trainable_flag = is_training
        if(i<=2):
          trainable_flag = False

        if output_stride is not None and current_stride == output_stride:
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          layer_stride = 1
          layer_rate = rate
          rate *= conv_def.stride
        else:
          layer_stride = conv_def.stride
          layer_rate = 1
          current_stride *= conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = end_point_base
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point,
                            trainable=trainable_flag) # 第一个卷积层使用来自image_net的参数不可变
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points
          print_shape(net)

        elif isinstance(conv_def, DepthSepConv):
          end_point = end_point_base + '_depthwise'

          # By passing filters=None
          # separable_conv2d produces only a depthwise convolution layer
          net = slim.separable_conv2d(net, None, conv_def.kernel,
                                      depth_multiplier=1,
                                      stride=layer_stride,
                                      rate=layer_rate,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point,
                                      trainable=trainable_flag)

          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points
          print_shape(net)

          end_point = end_point_base + '_pointwise'

          net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                            stride=1,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point,
                            trainable=trainable_flag)

          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points
          print_shape(net)
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)

class mobilenet(object):
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

    with tf.variable_scope(tf.get_variable_scope(), noreuse_list, reuse=is_reuse):
      with slim.arg_scope(mult_network.faster_rcnn_arg_scope()):
        return self.build_network(sess, mode,tag, anchor_scales, anchor_ratios, reuse=is_reuse, is_training=(mode == 'TRAIN'))

  def build_network(self, sess,mode,tag, anchor_scales, anchor_ratios, reuse=False, is_training=True):
    with tf.variable_scope('MobilenetV1', 'MobilenetV1', reuse=reuse) as scope:
      with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
        net,end_points =  mobilenet_v1_base(self._image, scope=scope, final_endpoint='Conv2d_%d_pointwise' % (BRANCH_START_IDX-1), is_training=is_training)
        #self._act_summaries.append(net)
        self._layers['head'] = net
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
    print('Get variable to restore For Mobilenet...')
    #print('You have variables : ')
    #for v in variables:
    #  print('------ %s shape %s ---------' % (v.name,v.get_shape()) )
    #print('\n\n')

    variables_in_branch = set() 
    for task_id, _ in enumerate(self._tasks):
      for net_idx in range(BRANCH_START_IDX,len(_CONV_DEFS)):
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_depthwise/depthwise_weights:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/beta:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_mean:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_variance:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_pointwise/weights:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/beta:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_mean:0' % (task_id,net_idx))
        variables_in_branch.add('MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_variance:0' % (task_id,net_idx))

    variables_to_restore = []
    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'MobilenetV1/Conv2d_0/weights:0':
        self._variables_to_fix[v.name] = v
        continue

      if v.name in variables_in_branch:
        self._variables_to_fix[v.name] = v
        continue

      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore
        

  def fix_variables(self, sess, pretrained_model):
    print('Fix Mobilenet layers..')
    with tf.variable_scope('Fix_MobilenetV1') as scope:
      with tf.device("/cpu:0"):
        # fix the MobilenetV1 issue from conv weights to fc weights
        # fix RGB to BGR
        conv_2d_0_mult = tf.get_variable("conv_2d_0_mult",[3, 3, 3, 32],trainable=False)
        restorer_conv_rgb = tf.train.Saver({"MobilenetV1/Conv2d_0/weights":conv_2d_0_mult})
        restorer_conv_rgb.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix['MobilenetV1/Conv2d_0/weights:0'], 
                            tf.reverse(conv_2d_0_mult, [2])))
        print('Conv_0 fixed')

        #pass

        # Fix the conv layer for branch
        local_vars = locals()
        for net_idx in range(BRANCH_START_IDX,len(_CONV_DEFS)):
          conv_shape = _CONV_SHAPES[net_idx]
          local_vars['Conv2d_%d_depthwise_depthwise_weights' % net_idx]         =  tf.get_variable('Conv2d_%d_depthwise_depthwise_weights' % net_idx, conv_shape.dep_w)
          local_vars['Conv2d_%d_depthwise_BatchNorm_beta' % net_idx]            =  tf.get_variable('Conv2d_%d_depthwise_BatchNorm_beta' % net_idx, conv_shape.dep_beta_mean_vari)
          local_vars['Conv2d_%d_depthwise_BatchNorm_moving_mean' % net_idx]     =  tf.get_variable('Conv2d_%d_depthwise_BatchNorm_moving_mean' % net_idx, conv_shape.dep_beta_mean_vari)
          local_vars['Conv2d_%d_depthwise_BatchNorm_moving_variance' % net_idx] =  tf.get_variable('Conv2d_%d_depthwise_BatchNorm_moving_variance' % net_idx, conv_shape.dep_beta_mean_vari)
          local_vars['Conv2d_%d_pointwise_weights' % net_idx]                   =  tf.get_variable('Conv2d_%d_pointwise_weights' % net_idx, conv_shape.point_w)
          local_vars['Conv2d_%d_pointwise_BatchNorm_beta' % net_idx]            =  tf.get_variable('Conv2d_%d_pointwise_BatchNorm_beta' % net_idx, conv_shape.point_beta_mean_vari)
          local_vars['Conv2d_%d_pointwise_BatchNorm_moving_mean' % net_idx]     =  tf.get_variable('Conv2d_%d_pointwise_BatchNorm_moving_mean' % net_idx, conv_shape.point_beta_mean_vari)
          local_vars['Conv2d_%d_pointwise_BatchNorm_moving_variance' % net_idx] =  tf.get_variable('Conv2d_%d_pointwise_BatchNorm_moving_variance' % net_idx, conv_shape.point_beta_mean_vari)

        # branch varaible to restore
        restore_branch_n2v = {}
        for net_idx in range(BRANCH_START_IDX,len(_CONV_DEFS)):
          restore_branch_n2v['MobilenetV1/Conv2d_%d_depthwise/depthwise_weights' % net_idx]         =  local_vars['Conv2d_%d_depthwise_depthwise_weights' % net_idx]        
          restore_branch_n2v['MobilenetV1/Conv2d_%d_depthwise/BatchNorm/beta' % net_idx]            =  local_vars['Conv2d_%d_depthwise_BatchNorm_beta' % net_idx]           
          restore_branch_n2v['MobilenetV1/Conv2d_%d_depthwise/BatchNorm/moving_mean' % net_idx]     =  local_vars['Conv2d_%d_depthwise_BatchNorm_moving_mean' % net_idx]             
          restore_branch_n2v['MobilenetV1/Conv2d_%d_depthwise/BatchNorm/moving_variance' % net_idx] =  local_vars['Conv2d_%d_depthwise_BatchNorm_moving_variance' % net_idx]
          restore_branch_n2v['MobilenetV1/Conv2d_%d_pointwise/weights' % net_idx]                   =  local_vars['Conv2d_%d_pointwise_weights' % net_idx]
          restore_branch_n2v['MobilenetV1/Conv2d_%d_pointwise/BatchNorm/beta' % net_idx]            =  local_vars['Conv2d_%d_pointwise_BatchNorm_beta' % net_idx]
          restore_branch_n2v['MobilenetV1/Conv2d_%d_pointwise/BatchNorm/moving_mean' % net_idx]     =  local_vars['Conv2d_%d_pointwise_BatchNorm_moving_mean' % net_idx]
          restore_branch_n2v['MobilenetV1/Conv2d_%d_pointwise/BatchNorm/moving_variance' % net_idx] =  local_vars['Conv2d_%d_pointwise_BatchNorm_moving_variance' % net_idx]

                                                                             
        restorer_conv_branch = tf.train.Saver(restore_branch_n2v)
        restorer_conv_branch.restore(sess, pretrained_model)

        for task_id , _ in enumerate(self._tasks):
          for net_idx in range(BRANCH_START_IDX,len(_CONV_DEFS)):
            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/depthwise_weights:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_depthwise_depthwise_weights' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/depthwise_weights:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/beta:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_depthwise_BatchNorm_beta' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/beta:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_mean:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_depthwise_BatchNorm_moving_mean' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_mean:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_variance:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_depthwise_BatchNorm_moving_variance' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_depthwise/BatchNorm/moving_variance:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/weights:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_pointwise_weights' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/weights:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/beta:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_pointwise_BatchNorm_beta' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/beta:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_mean:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_pointwise_BatchNorm_moving_mean' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_mean:0' % (task_id,net_idx)].get_shape() )))

            sess.run(tf.assign(self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_variance:0' % (task_id,net_idx)],
                                    tf.reshape(local_vars['Conv2d_%d_pointwise_BatchNorm_moving_variance' % net_idx],
                                        self._variables_to_fix['MobilenetV1/branch_%d/Conv2d_%d_pointwise/BatchNorm/moving_variance:0' % (task_id,net_idx)].get_shape() )))
        
        print('branch vars fixed')


