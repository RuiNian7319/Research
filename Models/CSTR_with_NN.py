import tensorflow as tf
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from CSTR_model import MimoCstr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

model = MimoCstr(nsim=50)
action_list = [-1, 1]


def reward_calc(temp, temp_sp):
    rewards = 0

    if temp_sp * 0.999 < temp < temp_sp * 1.001:
        rewards = rewards + 15 - abs(temp - temp_sp) * 20
    else:
        rewards = rewards - np.power(temp - temp_sp, 2)

    return rewards


num_inputs = 3
num_hidden = 4
num_output = 1   # Prob to go left            1 - left = right

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

output_layer = tf.layers.dense(hidden_layer_two, num_output, activation=tf.nn.sigmoid, kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1 - output_layer])
action = tf.multinomial(probabilities, num_samples=1)

init = tf.global_variables_initializer()

epi = 100
step_limit = 500
avg_steps = []

avg_reward = []

env = gym.make('CartPole-v0')


with tf.Session() as sess:
    sess.run(init)

    for i_episode in range(epi):

        # obs = env.reset()
        obs = model.reset()

        for step in range(1, model.Nsim + 1):

            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            action_picked = action_list[action_val[0][0]]                 #
            model.u[step, 0] = model.u[step - 1, 0] + action_picked                     #

            model.x[step, :] = model.next_state(model.x[step - 1, :], model.u[step, :])         #

            obs = model.x[step, :]                      #
            reward = reward_calc(model.x[step, 1], 324.5)             #

            # obs, reward, done, _ = env.step(action_val[0][0])

            if step == model.Nsim:

                # avg_steps.append(step)
                # print("Done after steps {}".format(step))

                avg_reward.append(reward)               #
                reward = 0                           #
                break

print("After {} episodes, the average steps was {}".format(epi, np.mean(avg_reward)))
env.close()






