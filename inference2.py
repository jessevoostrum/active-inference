import numpy as np
from scipy.special import softmax
import itertools


class Agent:

    def __init__(self, A, B, C, D=None, plan_num_steps_ahead=1):

        self.A = A
        self.B = B
        self.C = C

        num_states = B.shape[0]

        if D:
            self.belief_current_state = D
        else:
            self.belief_current_state = np.ones(num_states) / num_states

        num_actions = B.shape[2]

        self.policy_space = list(itertools.product(*[range(s) for s in [num_actions] * plan_num_steps_ahead]))


    def sample_action(self):
        efes = np.zeros(len(self.policy_space))
        for idx, policy in enumerate(self.policy_space):
            belief_next_states = get_belief_next_states(self.belief_current_state, self.B, policy)
            belief_next_observations = get_belief_next_observations(belief_next_states, self.A)
            efes[idx] = compute_expected_free_energy(belief_next_observations, self.C)

        q_pi = softmax(efes)

        sampled_policy = np.random.sample(self.policy_space, p=q_pi)

        return sampled_policy[0]

    def update_belief_current_state(self, observation, action):
        update_belief_current_state(observation, action, self.A, self.B, self.belief_current_state)


def update_belief_current_state(observation, action, A, B, prior):

    posterior_given_action = np.matmul(B[:, :, action], prior)

    posterior_given_action_observation = A[observation, :] * posterior_given_action

    posterior_given_action_observation /= np.sum(posterior_given_action_observation)

    return posterior_given_action_observation

def get_belief_next_states(belief_current_state, B, policy):

    belief_next_states = []

    belief_prev_state = belief_current_state

    for tau, action in enumerate(policy):
        belief_next_state = np.matmul(B[:, :, action], belief_prev_state)

        belief_next_states.append(belief_next_state)

        belief_prev_state = belief_next_state

    return belief_next_states

def get_belief_next_observations(belief_next_states, A):
    return [np.matmul(A, belief_next_state) for belief_next_state in belief_next_states]

def compute_expected_free_energy(belief_next_observations, C):

    utility = 0

    for belief_next_observation in belief_next_observations:
        utility += np.dot(belief_next_observation, C)

    return utility




