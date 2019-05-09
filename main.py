# import tensorflow as tf
#
# with tf.device('/cpu:0'):
#
#         a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
#
#         b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
#
# with tf.device('/gpu:1'):
#
#         c = a+b
#
# print(c)
#
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
#
# sess.run(tf.global_variables_initializer())
#
# print(sess.run(c))


import torch
print(torch.__version__)
print(torch.cuda.is_available())
