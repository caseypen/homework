import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    max_path_length = horizon or env.spec.max_episode_steps
    gamma = 0.99 
    """ YOUR CODE HERE """
    for itr in range(num_paths):
        print("********** Iteration %i ************"%itr)
        ob = env.reset()
        obs, acs, rewards, obs_2, returns = [], [], [], [], []
        animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and render)
        steps = 0
        g = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.05)
            obs.append(ob)
            ac = controller.get_action(ob)
            # print ac
            acs.append(ac)
            # print ac
            ob, rew, done, _ = env.step(ac)
            g += rew*gamma**steps
            obs_2.append(ob)
            rewards.append(rew)
            returns.append(g)
            steps += 1
            if done or steps > max_path_length:
                terminated = done
                break 
        path = {"observation" : np.array(obs), 
                "reward" : np.array(rewards), 
                "action" : np.array(acs),
                "obs_next" : np.array(obs_2),
                "return": np.array(returns)}
        paths.append(path)
    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def pathlength(path):
    return len(path["reward"])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    obs = np.concatenate([path["observation"] for path in data])
    action = np.concatenate([path["action"] for path in data])
    next_ob = np.concatenate([path["obs_next"] for path in data])

    deltas = obs - next_ob
    mean_obs = np.mean(obs)
    std_obs = np.std(obs)
    mean_deltas = np.mean(deltas)
    std_deltas = np.std(deltas)
    mean_action = np.mean(action)
    std_action = np.std(action)

    return mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    pass

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=10, 
         num_simulated_paths=1000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None,
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    #========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    """ YOUR CODE HERE """
    paths_rand = sample(env, random_controller, num_paths=num_paths_random, horizon=env_horizon,render=render,verbose=False)
    #========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    # 
    """ YOUR CODE HERE """
    normalization = compute_normalization(paths_rand)

    gamma = 0.99

    #========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    # 
        # prefit dynamic before on policy dagger:
    print("****** Pretrain dynamic Model *******")
    losses = []
    obs_rand = np.concatenate([path["observation"] for path in paths_rand])
    action_rand = np.concatenate([path["action"] for path in paths_rand])
    next_ob_rand = np.concatenate([path["obs_next"] for path in paths_rand])
    data_size_rand = obs_rand.shape[0]
    for i in range(1000):
        # obtain batch size from random policy
        batch_idx_rand = np.random.randint(data_size_rand, size = batch_size)
        batch_ob_rand = obs_rand[batch_idx_rand,:]
        batch_ac_rand = action_rand[batch_idx_rand,:]
        batch_nxt_rand = next_ob_rand[batch_idx_rand,:]
        # obtain batch size from on policy
        batch_ob = np.copy(batch_ob_rand)
        batch_ac = np.copy(batch_ac_rand)
        batch_nxt = np.copy(batch_nxt_rand)
        loss = dyn_model.fit(batch_ob,batch_ac,batch_nxt)
        losses.append(loss)
        if(i%20==0):
            print('loss', loss)

    costs = []
    returns = []
    paths_rl = []
    for itr in range(onpol_iters):
        """ YOUR CODE HERE """
        # fit dynamic model
        if itr > 0:
            obs_rl = np.concatenate([path["observation"] for path in paths_rl])
            action_rl = np.concatenate([path["action"] for path in paths_rl])
            next_ob_rl = np.concatenate([path["obs_next"] for path in paths_rl])
        obs_rand = np.concatenate([path["observation"] for path in paths_rand])
        action_rand = np.concatenate([path["action"] for path in paths_rand])
        next_ob_rand = np.concatenate([path["obs_next"] for path in paths_rand])
        # print obs[128,:].shape
        data_size_rand = obs_rand.shape[0]
        if itr > 0:
            data_size_rl = obs_rl.shape[0]
        # batch_size=128
        losses = []
        # fit model function
        for i in range(dynamics_iters):
            # obtain batch size from random policy
            batch_idx_rand = np.random.randint(data_size_rand, size = batch_size/20)
            batch_ob_rand = obs_rand[batch_idx_rand,:]
            batch_ac_rand = action_rand[batch_idx_rand,:]
            batch_nxt_rand = next_ob_rand[batch_idx_rand,:]
            # obtain batch size from on policy
            if itr > 0:
                batch_idx_rl   = np.random.randint(data_size_rl, size = batch_size*19/20)
                batch_ob_rl = obs_rl[batch_idx_rl,:]
                batch_ac_rl = action_rl[batch_idx_rl,:]
                batch_nxt_rl = next_ob_rl[batch_idx_rl,:]
                # mix them
                batch_ob = np.concatenate((batch_ob_rand,batch_ob_rl))
                batch_ac = np.concatenate((batch_ac_rand,batch_ac_rl))
                batch_nxt = np.concatenate((batch_nxt_rand,batch_nxt_rl))
            else:
                batch_ob = np.copy(batch_ob_rand)
                batch_ac = np.copy(batch_ac_rand)
                batch_nxt = np.copy(batch_nxt_rand)
            loss = dyn_model.fit(batch_ob,batch_ac,batch_nxt)
            losses.append(loss)
            # if(i%20==0):
            #     print('loss', loss)
        print("on policy dagger ", itr)
        ob = env.reset()
        observes, acs, rewards, obs_2, returns = [], [], [], [], []
        steps = 0
        g = 0
        max_path_length = mpc_controller.horizon
        timesteps_this_batch=0
        while True:
            while True:
                observes.append(ob)
                ac = mpc_controller.get_action(ob)
                # print ac
                acs.append(ac)
                # print ac
                ob, rew, done, _ = env.step(ac)
                g += rew*gamma**steps
                obs_2.append(ob)
                rewards.append(rew)
                returns.append(g)
                steps += 1
                if done or steps > max_path_length:
                    terminated = done
                    break 
            path = {"observation" : np.array(observes), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs),
                    "obs_next" : np.array(obs_2),
                    "return": np.array(returns)}
            paths_rl.append(path)
            timesteps_this_batch += pathlength(path)
            print g
            if timesteps_this_batch > batch_size:
                break
        trajectory_cost = trajectory_cost_fn(cheetah_cost_fn,path["observation"], path["action"],path["obs_next"])
        costs.append(trajectory_cost)
        returns.append(path["return"][-1])
        
        # print batch_ob.shape

        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=10)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths, 
                 num_paths_onpol=args.onpol_paths, 
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len, 
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    main()
