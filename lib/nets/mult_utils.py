import numpy as np
import tensorflow as tf   
from model.config import cfg
from mult_vgg16 import vgg16
from mult_mobilenet import mobilenet  

network_map = {
   'vgg16':vgg16,
   'mobilenet':mobilenet
}


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    #print 'grads'
    for g, var in grad_and_vars:
      if g is None:
        continue
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
      #print var.name, len(grad_and_vars)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads 
 
def create_train_op(sess, task_list, batch_size):
  with tf.device('/cpu:0'):
    lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
    momentum = cfg.TRAIN.MOMENTUM
    optimizer = tf.train.MomentumOptimizer(lr, momentum)
    tasks= []
    tower_grads = []
    tower_losses = []
    is_reuse = False
    net_name = 'vgg16'
    if(net_name not in network_map):
      raise ValueError('network name %s not konwn!' % net_name)
      
    for task_id , task_classes_num in enumerate(task_list):
      net = network_map[net_name](batch_size)
      with tf.device('/gpu:%d' % task_id):
        with tf.name_scope('tower_%d' % task_id) as scope:
          net.create_mult_architecture(sess, 'TRAIN', task_list, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS,
                                            is_reuse = is_reuse)
          #layers = mult_nets[task_id]
          task = net.get_task(task_id)
          loss = task['losses']['total_loss']
          #loss = layers['total_loss']
          tasks.append(task)
          # Compute the gradients wrt the loss
          gvs = optimizer.compute_gradients(loss)
          # Double the gradient of the bias if set
          if cfg.TRAIN.DOUBLE_BIAS:
            final_gvs = []
            with tf.variable_scope('Gradient_Mult') as scope:
              for grad, var in gvs:
                if grad is None:
                  final_gvs.append((grad, var))
                  continue
                scale = 1.
                if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                  scale *= 2.
                if not np.allclose(scale, 1.0):
                  grad = tf.multiply(grad, scale)
                final_gvs.append((grad, var))
            tower_grads.append(final_gvs)
          else:
            tower_grads.append(gvs)

          tower_losses.append(loss)
          is_reuse = True

    loss = tf.div(tf.add_n(tower_losses), 1.0 * len(tower_losses))
    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads)

    batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batch_norm_updates)
    train_op = tf.group(apply_gradient_op, #variables_averages_op,
                            batchnorm_updates_op)

    return tasks, train_op, lr, net

def train_step(sess, tasks, blobs_list, train_op):
  assert len(tasks) == len(blobs_list)
  feed_dict = {}
  op_list = []
  for task_id, task in enumerate(tasks):
    blobs = blobs_list[task_id]
    feed_dict[task['image_ph']] = blobs['data']
    feed_dict[task['im_info_ph']] = blobs['im_info']
    feed_dict[task['gt_boxes_ph']] = blobs['gt_boxes']
    losses = task['losses']
    op_list += [losses["rpn_cross_entropy"], losses["rpn_loss_box"], losses['cross_entropy'],\
                    losses["loss_box"], losses["total_loss"]]
  op_list.append(train_op)
  results = sess.run(op_list, feed_dict)

  loss_cnt = 5
  l1 =[]
  l2 =[]
  l3 =[]
  l4 =[]
  l5 =[] 
  for task_id, _ in enumerate(tasks):
    l1.append(results[task_id*loss_cnt])
    l2.append(results[task_id*loss_cnt + 1])
    l3.append(results[task_id*loss_cnt + 2])
    l4.append(results[task_id*loss_cnt + 3])
    l5.append(results[task_id*loss_cnt + 4])
  return l1, l2, l3, l4, l5
