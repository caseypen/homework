import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        # get information of environment
        # Observation and action sizes
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # placeholder for input state, action and next state
        self.obs = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.obs_nxt = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)

        self.iter = iterations
        self.bs = batch_size
        self.sess = sess
        # statistics of environment from random roll out
        self.mean_obs = normalization[0]
        self.std_obs = normalization[1]
        self.mean_deltas = normalization[2]
        self.std_deltas = normalization[3]
        self.mean_action = normalization[4]
        self.std_action = normalization[5]
        # normalize inputs of states and actions
        self.obs_normal = (self.obs-self.mean_obs)/(self.std_obs+1e-5)
        self.action_normal = (self.action - self.mean_action)/(self.std_action+1e-5)

        self.s_a_pairs = tf.concat([self.obs_normal, self.action_normal],axis=1, name="ob_ac_pairs")

        self.out = build_mlp(self.s_a_pairs, ob_dim, "state_est", 
                            n_layers=n_layers,
                            size=size, 
                            activation=activation,
                            output_activation=output_activation)
        # denormalize delta and output estimation of next state
        self.delta_s = self.mean_deltas + self.std_deltas*self.out
        self.obs_est = self.obs + self.delta_s

        self.loss = tf.nn.l2_loss(self.obs_est - self.obs_nxt, name='model_loss_state')
        self.trainer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, obs, action, next_ob):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        # unpackage dataset:
        # obs = np.concatenate([path["observation"] for path in data])
        # action = np.concatenate([path["action"] for path in data])
        # next_ob = np.concatenate([path["obs_next"] for path in data])

        loss, _ = self.sess.run([self.loss,self.trainer], feed_dict={ self.obs:obs,
                                           self.action: action,
                                           self.obs_nxt: next_ob})
        return loss
    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        # obs = np.concatenate([path["observation"] for path in data])
        # action = np.concatenate([path["action"] for path in data])
        # next_ob = np.concatenate([path["obs_next"] for path in data])

        return self.sess.run(self.obs_est, feed_dict={ self.obs:states,
                                                     self.action: actions})