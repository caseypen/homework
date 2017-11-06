import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.nn.tanh,
        output_activation=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        out = input_placeholder
        for i in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation, name="layer"+str(i))
        out = tf.layers.dense(out, output_size, activation=output_activation, name="output_layer")
    return out

def pathlength(path):
    return len(path["reward"])
# figure out log probability of sampled continuous action
def normal_log_prob(x, mean, log_std, dim):
    """
    x: [batch, dim]
    return: [batch]
    """
    zs = (x - mean) / tf.exp(log_std)
    return - tf.reduce_sum(log_std, axis=1) - \
           0.5 * tf.reduce_sum(tf.square(zs), axis=1) - \
           0.5 * dim * np.log(2 * np.pi)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             bootstrap=False
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print ob_dim, ac_dim
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="advantage", dtype=tf.float32)
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = build_mlp(sy_ob_no, 
                                 ac_dim, 
                                 "discrete_mlp", 
                                 n_layers=n_layers, 
                                 size=size, 
                                 activation=tf.nn.relu, 
                                 output_activation=None)
        # print sy_logits_na      
        sy_logprob_na = tf.nn.log_softmax(sy_logits_na)
        sy_sampled_ac = tf.multinomial(sy_logprob_na, 1) # Hint: Use the tf.multinomial op
        # print sy_sampled_ac
        batch_n = tf.shape(sy_ob_no)[0]
        act_index = tf.stack([tf.range(0,batch_n),sy_ac_na],axis=1)
        # sy_sampled_ac = tf.gather_nd(sy_sampled_ac,tf.range(0,batch_n))
        # sy_sampled_ac = sy_sampled_ac[0]
        sy_logprob_n = tf.gather_nd(sy_logprob_na, act_index)

    else:
        # YOUR_CODE_HERE
        sy_mean = build_mlp( sy_ob_no, 
                             ac_dim, 
                             "continuous_mlp", 
                             n_layers=2, 
                             size=32, 
                             activation= tf.nn.relu, 
                             output_activation= None)
        sy_logstd = tf.Variable(tf.ones(batch_n),name="std") # logstd should just be a trainable variable, not a network output.
        sy_sampled_ac = sy_mean + sy_logstd*tf.random_normal(tf.shape(sy_mean))
        sy_logprob_n =  normal_log_prob(sy_ac_na, sy_mean, sy_log_std, ac_dim)# Hint: Use the log probability under a multivariate gaussian. 



    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    loss =  -tf.reduce_mean(sy_logprob_n*sy_adv_n)# Loss function that we'll differentiate to get the policy gradient.
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1, 
                                "nn_baseline",
                                n_layers=1,
                                size=32,
                                activation=tf.nn.relu,
                                output_activation=None))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        # YOUR_CODE_HERE
        v_t = tf.placeholder("float",[None])
        l_2 = 0.5*tf.nn.l2_loss(v_t - baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(l_2)


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards, obs_2 = [], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                pi = sess.run(sy_logits_na, feed_dict={sy_ob_no: ob[None]})
                # print pi
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                # print ac
                ac = ac[0][0]
                # print ac
                acs.append(ac)
                # print ac
                ob, rew, done, _ = env.step(ac)
                obs_2.append(ob)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    terminated = done
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs),
                    "obs_next" : np.array(obs_2)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        ob_next_no = np.concatenate([path["obs_next"] for path in paths])
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#
        
        # q_n = np.zero(q_n.shape)
        # YOUR_CODE_HERE
        if reward_to_go:
            q_n = []
            # for path in paths.reverse():
            #     q_t = 0 
            #     r_path = path["reward"].reverse()
            #     path_len = pathlength(r_path)
            #     for r in enumerate(r_path):
            #         q_t = r + gamma*q_t
            #         q_n[i] = q_t
            #         i += 1
            # q_n.reverse()
            if not bootstrap:
                for path in paths:
                    rew_t = path["reward"]
                    return_t=discount(rew_t, gamma)
                    q_n.append(return_t)
            else:
                for path in paths:
                    v_nxt = sess.run(baseline_prediction, feed_dict={sy_ob_no: path["obs_next"]})
                    q_target = v_nxt + path["reward"]
                    q_n.append(q_target)
            q_n = np.concatenate(q_n)            
        else:
            i = 0
            q_n = np.concatenate([path["reward"] for path in paths])
            for path in paths:
                q_t = 0
                for idx, r in enumerate(path["reward"]):
                    q_t += gamma**idx * r
                    q_n[i] = q_t
                    i += 1
                    

                

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)
            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no : ob_no})
            adv_n = q_n - b_n
        else:          
            adv_n = q_n.copy()
            # print q_n

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            normal_adv = tf.nn.l2_normalize(sy_adv_n,0, epsilon=1e-8, name="adv_normal")
            sess.run(normal_adv, feed_dict={sy_adv_n:adv_n})

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            v_target = []
            for path in paths:
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                v_target.append(return_t)
            v_target = np.concatenate(v_target)
            print v_target.shape
            for _ in range(40):
                sess.run(baseline_update_op, feed_dict={sy_ob_no: ob_no,
                                                        v_t: v_target})
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        sess.run(update_op, feed_dict={ sy_ob_no:ob_no,
                                        sy_ac_na:ac_na,
                                        sy_adv_n:adv_n})

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', '-dc', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--bootstrap', '-bts', action='store_true')
    args = parser.parse_args()
    # config = tf.ConfigProto( device_count = {'CPU': 0})
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    def train_func(seed):
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                bootstrap=args.bootstrap)
    training_threads=[]
    # arg_tuple=[args.exp_name,
    #             args.env_name,
    #             args.n_iter,
    #             args.discount,
    #             args.batch_size,
    #             max_path_length,
    #             args.learning_rate,
    #             args.reward_to_go,
    #             args.render,
    #             os.path.join(logdir,'%d'%args.seed),
    #             not(args.dont_normalize_advantages),
    #             args.nn_baseline, 
    #             1,
    #             args.n_layers,
    #             args.size,
    #             args.bootstrap]
    # arg_tuple=list(arg_tuple)
    with tf.device('/cpu:0'):
        for e in range(args.n_experiments):
            seed = args.seed + 10*e
            print('Running experiment with seed %d'%seed)
            # arg_tuple[9]=os.path.join(logdir,'%d'%seed)
            # arg_tuple[12]=seed
            # arg_tuple=tuple(arg_tuple)
            # def train_func():
                # train_PG(
                #     exp_name=args.exp_name,
                #     env_name=args.env_name,
                #     n_iter=args.n_iter,
                #     gamma=args.discount,
                #     min_timesteps_per_batch=args.batch_size,
                #     max_path_length=max_path_length,
                #     learning_rate=args.learning_rate,
                #     reward_to_go=args.reward_to_go,
                #     animate=args.render,
                #     logdir=os.path.join(logdir,'%d'%seed),
                #     normalize_advantages=not(args.dont_normalize_advantages),
                #     nn_baseline=args.nn_baseline, 
                #     seed=seed,
                #     n_layers=args.n_layers,
                #     size=args.size,
                #     bootstrap=args.bootstrap)
            training_thread = Process(target=train_func, args=(seed,))
            training_threads.append(training_thread)
            # Awkward hacky process runs, because Tensorflow does not like
            # repeatedly calling train_PG in the same thread.
            # p = Process(target=train_func, args=tuple())
            # p.start()
            # p.join()
        for p in training_threads:
            p.start()
        for p in training_threads:
            p.join()
        

if __name__ == "__main__":
    main()
