import numpy as np
from pymdp.envs import TMazeEnv

from agent import Agent


reward_probabilities = [0.98, 0.02]  # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs=reward_probabilities)

A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()

C = [np.zeros(A_gp_i.shape[0]) for A_gp_i in A_gp]
C[1][0] = 2
C[1][1] = 3
C[1][2] = 1

T = 3  # number of time steps

agent = Agent(A_unflat=A_gp, B_unflat=B_gp, C_unflat=C, plan_num_steps_ahead=T-1)

obs = env.reset()  # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
print(msg.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

action = None

for t in range(T):

    agent.update_belief_current_state(obs, action)

    action = agent.sample_action()

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [{},  {}, {}]"""
    print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))



