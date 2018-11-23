import tensorflow as tf
import numpy as np

a = tf.constant([1, 3])
b = tf.constant([5, 3])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    c = tf.add(a, b)
    print('{}'.format(sess.run(c)))


