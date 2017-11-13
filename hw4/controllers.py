import numpy as np
from cost_functions import trajectory_cost_fn
from cost_functions import cheetah_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.ac_space = env.action_space # obtain action dimension

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.ac_space.sample()		


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		states = []
		actions = []
		nxt_states = []
		state_batch = np.tile(state, [self.num_simulated_paths, 1]) # copy state for simulated numbers path and make them a fake batch
		# do the imagination rolling out
		for j in range(self.horizon):
			action_batch = []
			for i in range(self.num_simulated_paths): # random policy 
				action_batch.append(self.env.action_space.sample()) # choose random actions for next step of 10 simulated paths 
			action_batch = np.asarray(action_batch)
			nxt_batch = self.dyn_model.predict(state_batch,action_batch) # rollout next batches
			
			states.append(state_batch)
			actions.append(action_batch)
			nxt_states.append(nxt_batch)
			state_batch = np.copy(nxt_batch)
		
		traj_s=np.asarray(states).transpose(1,0,2)
		traj_nxt_s = np.asarray(nxt_states).transpose(1,0,2)
		traj_action = np.asarray(actions).transpose(1,0,2)
		costs = []
		for i in range(traj_s.shape[0]):
		    trajectory_cost = trajectory_cost_fn(cheetah_cost_fn,traj_s[i], traj_action[i],traj_nxt_s[i])
		    costs.append(trajectory_cost)

		a_idx = np.argmin(np.array(costs))

		return traj_action[a_idx][0]