
import numpy as np

import environment

from agent import Agent

# np.random.seed(80801)

C = np.array([5, 10, -2, 10, -2, 10, -2, 10])

agent = Agent(C=C, num_obs_per_dim=[8], num_states_per_dim=[2,2,2], num_actions=4, plan_num_steps_ahead=5)

env = environment.Agent(0)
env.reset(0)

T = 2000  # number of time steps

for t in range(T):

    action = agent.sample_action()
    print("action", action)

    if action == 0:
        actions = np.array([0, 0])
    if action == 1:
        actions = np.array([0, 1])
    if action == 2:
        actions = np.array([1, 0])
    if action == 3:
        actions = np.array([1, 1])

    obs_binary, internal_env_states = env.step(actions, iteration=t, sampling =True, plot=True, trajectorie=False)

    obs_binary_list = ''.join([str(int(item)) for item in obs_binary])
    obs = int(obs_binary_list, 2)

    agent.update_belief_current_state(obs, action)

    agent.update_A(obs)

    agent.update_B()

    print(agent.belief_current_state, agent.belief_last_state)



















