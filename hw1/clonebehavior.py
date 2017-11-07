#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import tflearn
# ===========================
#   Actor and Critic DNNs
# ===========================

class CloneBehavior(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, env, learning_rate):
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.sess = sess
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.lr = learning_rate

        # Actor Network
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.batch_size = tf.shape(self.sy_ob_no)[0]
        self.sy_agent_pi = self.create_actor_network()
        self.sy_expert_pi = tf.placeholder(shape=[None, self.ac_dim], name="expert_pi", dtype=tf.float32)

        # self.loss = 0.5*tf.nn.l2_loss(self.sy_agent_pi - self.sy_expert_pi)
        self.loss = tf.reduce_mean(tf.square(self.sy_agent_pi - self.sy_expert_pi))
        # self.loss = tflearn.mean_square(self.sy_agent_pi, self.sy_expert_pi)
        self.trainer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def create_actor_network(self):
        h_1 = tf.layers.dense(self.sy_ob_no,128,activation=None,name="hidden_layer_1")
        h_2 = tf.layers.dense(h_1,64,activation=None,name="hidden_layer_2")
        h_3 = tf.layers.dense(h_2,64,activation=tf.nn.relu,name="hidden_layer_3")
        logits_na = tf.layers.dense(h_3,self.ac_dim,activation=None,name="logits")
        return logits_na
        # net = tflearn.fully_connected(self.sy_ob_no, 128, activation=None)
        # net = tflearn.layers.normalization.batch_normalization(net)
        # # action = tflearn.input_data(shape=[None,self.a_dim]) # not take in action when calculating Q 
        # net = tflearn.conv_2d(inputs, 8, 3, activation='relu', name='conv1') # inputs must be 4D tensor
        # # net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv2')
        # net = tflearn.fully_connected(inputs, 64, activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)

        # # Add the action tensor in the 2nd hidden layer
        # # Use two temp layers to get the corresponding weights and biases
        # # t1 = tflearn.fully_connected(net, 64)
        # # t2 = tflearn.fully_connected(action, 64)

        # # net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # net = tflearn.fully_connected(net, 64, activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.fully_connected(net, 32, activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)
        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(net, self.ac_dim)
        # return out

    def train(self, ob_no, expert_pi):
        return self.sess.run([self.trainer, self.loss], feed_dict={
            self.sy_ob_no: ob_no,
            self.sy_expert_pi: expert_pi 
        })
        return self.loss

    def predict(self, inputs):
        return self.sess.run(self.sy_agent_pi, feed_dict={
            self.sy_ob_no: inputs
        })

def train(cb_agent, sess, observations, actions):
    cb_agent.train(observations,actions)
def test(cb_agent, env, render):
    returns = []
    for i in range(20):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = cb_agent.predict(obs[None])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
            # if steps >= 2000:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of expert roll outs')
    parser.add_argument('--batch_size', '-bs', type=int, default=500)
    parser.add_argument("--Dagger",action='store_true')
    parser.add_argument("--learning_rate",'-lr',type=float, default=1e-4)
    args = parser.parse_args()
    Dagger = args.Dagger
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    render = args.render
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    import gym
    env = gym.make(args.envname)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print ob_dim, ac_dim
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            cb_agent=CloneBehavior(sess, env, learning_rate)
            sess.run(tf.global_variables_initializer())
            # tf_util.initialize()
            max_steps = args.max_timesteps or env.spec.timestep_limit
            # train from imitating learning without dagger
            paths = []
            path = {"obs":np.load("./expert_obs.npy"),
                    "acts":np.load("./expert_actions.npy")}
            print("expert obs data loaded: ", path['obs'].shape)
            print("expert act data loaded: ", path['acts'].shape)
            paths.append(path)
            # aggregate with Dagger
            if Dagger:
                print("Using Dagger to improve!")
            for i in range(args.num_rollouts):
                print("********** Iteration %i ************"%i) 
                np.random.seed(i)  
                # training with expert exploration
                ob_no =  np.concatenate([path["obs"] for path in paths])
                ac_no = np.concatenate([path["acts"] for path in paths])
                data_size = ob_no.shape[0]
                print data_size
                for j in range(500):
                    batch_idx = np.random.randint(data_size, size = batch_size)
                    batch_ob = ob_no[batch_idx,:]
                    batch_ac = ac_no[batch_idx,:]
                    _,loss = cb_agent.train(batch_ob,batch_ac)
                print('loss', loss)
                
                # Dagger to improve with expert label
                for k in range(20):
                    # print(" ## Rollout %i"%k)
                    obs = env.reset()
                    done = False
                    totalr = 0
                    steps = 0 # keep batch size
                    loss_mean=0
                    observations, expert_actions, returns, path = [], [], [], []
                    while not done:
                        expert_action = policy_fn(obs[None])
                        agent_action = cb_agent.predict(obs[None])
                        observations.append(obs)
                        expert_actions.append(expert_action[0])
                        if Dagger:
                            obs, r, done, _ = env.step(agent_action)
                        totalr += r
                        steps += 1
                        if args.render:
                            env.render()
                    returns.append(totalr) # return in an episode
                    path = {"obs":np.array(observations),
                            "acts":np.array(expert_actions)}
                    # print path["obs"].shape
                    # print path["acts"].shape
                    paths.append(path)
                print('mean return', np.mean(returns))
            # test(cb_agent, env, i, batch_size)
if __name__ == '__main__':
    main()
