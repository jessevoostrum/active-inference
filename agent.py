""" Agent class
The numbers in the comments of the methods refer to equation numbers in the paper:
"A Concise Mathematical Description of Active Inference in Discrete Time",
by Jesse van Oostrum, Carlotta Langer and Nihat Ay
"""

import numpy as np
from scipy.special import softmax
import itertools
from flattener import Flattener


class Agent:

    def __init__(self, A, B, C, D=None, plan_num_steps_ahead=1):

        self.flattener = Flattener(A, B, C, D)

        self.A = self.flattener.A_flat
        self.B = self.flattener.B_flat
        self.C = softmax(self.flattener.C_flat)
        self.D = self.flattener.D_flat

        self.pA = np.ones(self.A.shape)
        self.pB = np.ones(self.B.shape)

        self.belief_current_state = self.get_belief_initial_state()
        self.belief_last_state = self.belief_current_state

        self.last_action = None

        self.policy_space = self.make_policy_space(num_actions=self.B.shape[2], plan_num_steps_ahead=plan_num_steps_ahead)


        self.epsilon = 0.0000001  # avoid division by zero, logs of zero


    def sample_action(self):
        """(1)"""
        efes = np.zeros(len(self.policy_space))
        for idx, policy in enumerate(self.policy_space):
            belief_next_states = self.get_belief_next_states(policy)
            efes[idx] = self.compute_expected_free_energy(belief_next_states,
                                                          self.get_belief_next_observations(belief_next_states),
                                                          self.get_belief_next_states_given_observation(belief_next_states))

        q_pi = softmax(efes)
        sampled_policy = self.policy_space[np.random.choice(len(self.policy_space), p=q_pi)]
        action = sampled_policy[0]
        self.last_action = action
        action_unflat = self.flattener.unflatten_action(action)
        return action_unflat

    def compute_expected_free_energy(self, belief_next_states, belief_next_observations, belief_next_states_given_observation):
        """(4) and (5)"""
        utility = self.compute_utility(belief_next_observations)

        info_gain = self.compute_info_gain(belief_next_states, belief_next_observations, belief_next_states_given_observation)

        efe = utility + info_gain
        return efe

    def compute_info_gain(self, belief_next_states, belief_next_observations, belief_next_states_given_observation):
        """Sum over tau of the first term on the RHS of (5)"""
        info_gain = 0
        for belief_next_state, belief_next_observations, belief_next_state_given_observation in zip(belief_next_states, belief_next_observations, belief_next_states_given_observation):

            KL_div = belief_next_state_given_observation * np.log((belief_next_state_given_observation + self.epsilon)/ (np.array([belief_next_state]).T + self.epsilon))
            KL_div = KL_div.sum(axis=0)

            expectation = np.dot(belief_next_observations, KL_div)

            info_gain += expectation

        return info_gain

    def compute_utility(self, belief_next_observations):
        """Sum over tau of the second term on the RHS of (5)"""
        utility = 0
        for belief_next_observation in belief_next_observations:
            utility += np.dot(belief_next_observation, np.log(self.C + self.epsilon))
        return utility

    def update_belief_current_state(self, observation_unflat, action_unflat=None):
        """(6)"""
        self.belief_last_state = self.belief_current_state

        observation = self.flattener.flatten_observation(observation_unflat)

        if action_unflat is None:
            prior = self.belief_current_state
        else:
            action = self.flattener.flatten_action(action_unflat)
            prior = np.matmul(self.B[:, :, action], self.belief_current_state)

        posterior = prior * self.A[observation, :]

        posterior /= np.sum(posterior) + self.epsilon

        self.belief_current_state = posterior

    def get_belief_next_states(self, policy):
        """(7)"""
        belief_next_states = []
        belief_prev_state = self.belief_current_state

        for tau, action in enumerate(policy):
            belief_next_state = np.matmul(self.B[:, :, action], belief_prev_state)

            belief_next_states.append(belief_next_state)

            belief_prev_state = belief_next_state

        return belief_next_states

    def get_belief_next_states_given_observation(self, belief_next_states):
        """ (8)
        list of matrices p(s_\tau|o_\tau). dims: [time][state][observation] """

        belief_next_states_given_observation = []

        for belief_next_state in belief_next_states:
            belief_unnormalized = ((self.A + self.epsilon) * belief_next_state).T
            belief = belief_unnormalized / belief_unnormalized.sum(axis=0)
            belief_next_states_given_observation.append(belief)

        return belief_next_states_given_observation

    def get_belief_next_observations(self, belief_next_states):
        """(9)"""
        return [np.matmul(self.A, belief_next_state) for belief_next_state in belief_next_states]


    def get_belief_initial_state(self):
        if self.D:
            return  self.flattener.D_flat
        else:
            return self.uniform_distribution(num_states=self.B.shape[0])

    def reset_belief_state(self):
        self.belief_current_state = self.get_belief_initial_state()
        self.belief_last_state = self.belief_current_state

    # Learning
    def update_A(self, observation):
        """(21)"""
        self.update_pA(observation)
        self.A = self.pA / self.pA.sum(axis=0)

    def update_pA(self, observation):
        """(19)"""
        observation = self.flattener.flatten_observation(observation)
        self.pA[observation, :] += self.belief_current_state

    def update_B(self):
        """(21)"""
        self.update_pB()
        self.B = self.pB / self.pB.sum(axis=0)

    def update_pB(self):
        """(20)"""
        self.pB[:, :, self.last_action] += np.outer(self.belief_current_state, self.belief_last_state)


    @staticmethod
    def make_policy_space(num_actions, plan_num_steps_ahead):
        return list(itertools.product(*[range(s) for s in [num_actions] * plan_num_steps_ahead]))

    @staticmethod
    def uniform_distribution(num_states):
        return np.ones(num_states) / num_states








