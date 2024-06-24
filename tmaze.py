import numpy as np
from pymdp.envs import TMazeEnv

from agent import Agent


reward_probabilities = [0.98, 0.02]  # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs=reward_probabilities)

A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()

num_obs_per_modality = [A_gp_i.shape[0] for A_gp_i in A_gp]

C = [np.zeros(num_obs_modality_i) for num_obs_modality_i in num_obs_per_modality]
C[1][0] = 2
C[1][1] = 3
C[1][2] = 1

T = 3  # number of time steps

agent = Agent(A=A_gp, B=B_gp, C=C, plan_num_steps_ahead=T-1)


# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg1 = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
msg2 = """[Step {}] Action: [Move to {}]"""
msg3 = """[Step {}] Observation: [{},  {}, {}]"""

for _ in range(100):
    obs = env.reset()  # reset the environment and get an initial observation
    print(
        msg1.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]],
                    cue_observations[obs[2]]))

    action = None

    for t in range(T):

        agent.update_belief_current_state(obs, action)

        agent.update_A(obs)

        if t > 0:
            agent.update_B()

        action = agent.sample_action()

        print(msg2.format(t, location_observations[int(action[0])]))

        obs = env.step(action)

        print(msg3.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))



