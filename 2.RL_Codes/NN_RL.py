import numpy as np
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from CSTR_model import MimoCstr
from RL_Module import ReinforceLearning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def reward_calc(temp, temp_sp):
    rewards = 0

    if temp_sp * 0.997 < temp < temp_sp * 1.003:
        rewards = rewards + 15 - abs(temp - temp_sp) * 20
    else:
        rewards = rewards - np.power(temp - temp_sp, 2)

    return rewards


model = MimoCstr(delta=1, nsim=500)

rl = ReinforceLearning(discount_factor=0.97, states_start=300, states_stop=340, states_interval=0.5,
                       actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.1,
                       epsilon=0.1, doe=0, eval_period=5)

states = np.zeros([75])
states[0:15] = np.arange(290, 310, 20/15)
states[15:60] = np.arange(311, 330, 19/45)
states[60:75] = np.arange(331, 350, 19/15)

rl.user_states(list(states))

actions = np.zeros([20])
actions[1:20] = np.arange(-10, 10, 20/19)

rl.user_actions(list(actions))

num_of_episodes = 1

tf.reset_default_graph()

inputs = tf.placeholder(shape=[1, len(states)], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([len(states), len(actions)], 0, 0.01))
Qout = tf.matmul(inputs, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, len(actions)], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=rl.learning_rate)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

rList = []

with tf.Session() as sess:

    sess.run(init)
    saver.restore(sess, "/tmp/model.ckpt")

    for i in range(num_of_episodes):

        model.reset(random_init=True)
        rAll = 0

        for j in range(1, model.Nsim + 1):

            if j % rl.eval_period == 0:
                s = rl.state_detection(model.x[j - 1, 1])

                a, allQ = sess.run([predict, Qout], feed_dict={inputs: np.identity(len(states))[s:s + 1]})

                number = np.random.rand()
                if number < 0.0:
                    a = [np.random.randint(0, len(actions))]

                model.u[j, 0] = model.u[j - 1, 0] + rl.actions[a[0]]

                rl.feedback_evaluation(j)

            else:
                model.u[j, :] = model.u[j - 1, :]

            model.x[j, :] = model.next_state(model.x[j - 1, :], model.u[j, :])

            if j == rl.eval_feedback:
                s1 = rl.state_detection(model.x[j, 1])
                r = reward_calc(model.x[j, 1], 324.5)
                Q1 = sess.run(Qout, feed_dict={inputs: np.identity(len(states))[s1:s1 + 1]})
                maxQ1 = np.argmax(Q1)
                targetQ = allQ
                targetQ[0, a] = r + rl.discount_factor * maxQ1

                _, W1 = sess.run([updateModel, W], feed_dict={inputs: np.identity(len(states))[s:s+1], nextQ: targetQ})

                rAll += r

        rList.append(rAll)

    # Save model in a magical temporary place that doesn't physically exist...
    # save_path = saver.save(sess, "/tmp/model.ckpt")
    # print("Model saved in path: %s" % save_path)
