import numpy as np
from scipy.special import softmax
import itertools


class Agent:

    def __init__(self, C, A=None, B=None, num_obs_per_dim=None, num_states_per_dim=None, num_actions=None,
                 plan_num_steps_ahead=1):

        self.C = softmax(C)

        if A and B:
            self.A = A
            self.B = B

        else:
            self.num_states_per_dim = num_states_per_dim
            self.multi_idx_list_states = self.get_multi_idx_list(num_states_per_dim)
            self.multi_idx_list_obs = self.get_multi_idx_list(num_obs_per_dim)

            self.num_obs = len(self.multi_idx_list_obs)
            self.num_states = len(self.multi_idx_list_states)
            self.num_factors = len(num_states_per_dim)
            self.num_actions = num_actions

            self.pA = np.ones((self.num_obs, self.num_states)) + np.random.normal(0, .1, (self.num_obs, self.num_states))
            self.A = np.zeros((self.num_obs, self.num_states))
            self.update_A()

            # self.pAs = [np.ones((num_obs_dim, self.num_states)) + np.random.normal(0, .1, (num_obs_dim, self.num_states)) for num_obs_dim in num_obs_per_dim]
            self.pBs = [np.ones((num_states_dim, self.num_states, num_actions)) + np.random.normal(0, .1, (num_states_dim, self.num_states, num_actions)) for num_states_dim in num_states_per_dim]

            self.Bs = []
            self.update_Bs()

            self.B = self.flatten_Bs()

        self.belief_current_state = self.uniform_distribution(num_states=self.num_states)

        self.belief_last_state = self.belief_current_state

        self.last_action = None

        self.policy_space = self.make_policy_space(num_actions=self.B.shape[2], plan_num_steps_ahead=plan_num_steps_ahead)

        self.sample_maximum = True

        self.epsilon = 0.0000001  # avoid division by zero, logs of zero


    def update_B(self):
        self.update_pBs()
        self.update_Bs()
        self.B = self.flatten_Bs()

    def flatten_Bs(self):
        B = np.zeros((self.num_states, self.num_states, self.num_actions))

        for idx_action in range(self.num_actions):

            for single_idx_state_tau, multi_idx_state_tau in enumerate(self.multi_idx_list_states):
                for single_idx_state_tau_prev in range(self.num_states):
                    list_probs = [self.Bs[f][multi_idx_state_tau[f], single_idx_state_tau_prev, idx_action] for f in
                                  range(self.num_factors)]
                    joint_prob = self.product_of_elements(list_probs)

                    B[single_idx_state_tau, single_idx_state_tau_prev, idx_action] = joint_prob

        return B

    def update_A(self, observation=None):
        if observation:
            self.update_pA(observation)
        self.A = self.pA / self.pA.sum(axis=0)

    def update_pA(self, observation):
        self.pA[observation, :] += self.belief_current_state

    def update_Bs(self):
        self.Bs = []
        for pB in self.pBs:
             self.Bs.append(pB / pB.sum(axis=0))
    def update_pBs(self):
        for i, pB in enumerate(self.pBs):
            marginal_belief_current_state = self.marginalize_belief(self.belief_current_state, i)
            self.pBs[i][:, :, self.last_action] += np.outer(marginal_belief_current_state, self.belief_last_state)

    def marginalize_belief(self, joint_belief, dim):
        num_states_dim = self.num_states_per_dim[dim]
        marginal_belief = np.zeros(num_states_dim)
        for state in range(num_states_dim):
            for single_idx_state, multi_idx_state in enumerate(self.multi_idx_list_states):
                if multi_idx_state[dim] == state:
                    marginal_belief[state] += joint_belief[single_idx_state]
        return marginal_belief

    def sample_action(self):
        efes = np.zeros(len(self.policy_space))
        for idx, policy in enumerate(self.policy_space):
            belief_next_states = self.get_belief_next_states(policy)
            efes[idx] = self.compute_expected_free_energy(belief_next_states,
                                                          self.get_belief_next_observations(belief_next_states),
                                                          self.get_belief_next_states_given_observation(belief_next_states))

        q_pi = softmax(efes)

        if self.sample_maximum:
            sampled_policy = self.policy_space[efes.argmax()]
        else:
            sampled_policy = self.policy_space[np.random.choice(len(self.policy_space), p=q_pi)]

        action = sampled_policy[0]

        self.last_action = action

        return action

    def update_belief_current_state(self, observation, action=None):
        self.belief_last_state = self.belief_current_state

        if action is None:
            prior = self.belief_current_state
        else:
            prior = np.matmul(self.B[:, :, action], self.belief_current_state)

        posterior = prior * self.A[observation, :]

        posterior /= np.sum(posterior) + self.epsilon

        self.belief_current_state = posterior

    def get_belief_next_states(self, policy):

        belief_next_states = []
        belief_prev_state = self.belief_current_state

        for tau, action in enumerate(policy):
            belief_next_state = np.matmul(self.B[:, :, action], belief_prev_state)

            belief_next_states.append(belief_next_state)

            belief_prev_state = belief_next_state

        return belief_next_states

    def get_belief_next_observations(self, belief_next_states):
        return [np.matmul(self.A, belief_next_state) for belief_next_state in belief_next_states]

    def get_belief_next_states_given_observation(self, belief_next_states):
        """list of matrices p(s_\tau|o_\tau). dims: [time][state][observation] """

        belief_next_states_given_observation = []

        for belief_next_state in belief_next_states:
            belief_unnormalized = ((self.A + self.epsilon) * belief_next_state).T
            belief = belief_unnormalized / belief_unnormalized.sum(axis=0)
            belief_next_states_given_observation.append(belief)

        return belief_next_states_given_observation

    def compute_expected_free_energy(self, belief_next_states, belief_next_observations, belief_next_states_given_observation):

        utility = self.compute_utility(belief_next_observations)

        info_gain = self.compute_info_gain(belief_next_states, belief_next_observations, belief_next_states_given_observation)

        efe = utility + info_gain
        return efe

    def compute_utility(self, belief_next_observations):
        utility = 0
        for belief_next_observation in belief_next_observations:
            utility += np.dot(belief_next_observation, np.log(self.C + self.epsilon))
        return utility

    def compute_info_gain(self, belief_next_states, belief_next_observations, belief_next_states_given_observation):

        info_gain = 0

        for belief_next_state, belief_next_observations, belief_next_state_given_observation in zip(belief_next_states, belief_next_observations, belief_next_states_given_observation):

            KL_div = belief_next_state_given_observation * np.log((belief_next_state_given_observation + self.epsilon)/ (np.array([belief_next_state]).T + self.epsilon))
            KL_div = KL_div.sum(axis=0)

            expectation = np.dot(belief_next_observations, KL_div)

            info_gain += expectation

        return info_gain

    def get_multi_idx_list(self, num_elements_per_dim):
        multi_idx_list = list(itertools.product(*[range(s) for s in num_elements_per_dim]))
        return multi_idx_list

    @staticmethod
    def make_policy_space(num_actions, plan_num_steps_ahead):
        return list(itertools.product(*[range(s) for s in [num_actions] * plan_num_steps_ahead]))

    @staticmethod
    def uniform_distribution(num_states):
        return np.ones(num_states) / num_states

    @staticmethod
    def product_of_elements(lst):
        result = 1
        for num in lst:
            result *= num
        return result









