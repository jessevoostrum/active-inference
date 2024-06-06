import numpy as np
from scipy.special import softmax
import itertools


class Agent:

    def __init__(self, A_unflat, B_unflat, C_unflat, D_unflat=None, plan_num_steps_ahead=1):

        self.multi_idx_list_states = get_multi_idx_list_states(B_unflat)
        self.multi_idx_list_observations = get_multi_idx_list_observations(A_unflat)

        self.A = make_A_flat(A_unflat, self.multi_idx_list_observations)
        self.B = make_B_flat(B_unflat, self.multi_idx_list_states)

        self.C = softmax(make_C_flat(C_unflat, self.multi_idx_list_observations))

        num_states = self.B.shape[0]

        if D_unflat:
            self.belief_current_state = make_D_flat(D_unflat, self.multi_idx_list_states)
        else:
            self.belief_current_state = np.ones(num_states) / num_states

        num_actions = self.B.shape[2]

        self.policy_space = list(itertools.product(*[range(s) for s in [num_actions] * plan_num_steps_ahead]))

        self.epsilon = 0.0000001  # avoid division by zero, logs of zero

        self.num_factors = len(A_unflat[0].shape) - 1
    def sample_action(self):
        efes = np.zeros(len(self.policy_space))
        for idx, policy in enumerate(self.policy_space):
            belief_next_states = self.get_belief_next_states(policy)
            belief_next_observations = self.get_belief_next_observations(belief_next_states)
            belief_next_states_given_observation = self.get_belief_next_states_given_observation(belief_next_states)
            efes[idx] = self.compute_expected_free_energy(belief_next_states, belief_next_observations, belief_next_states_given_observation)

        # for efe, policy in zip(efes, self.policy_space):
        #     print(efe, policy)

        q_pi = softmax(efes)

        for q_pi_i, policy in zip(q_pi, self.policy_space):
            print(q_pi_i, policy)

        sampled_policy = self.policy_space[np.random.choice(len(self.policy_space), p=q_pi)]
        sampled_policy = self.policy_space[efes.argmax()]

        action = sampled_policy[0]
        action_unflat = make_action_unflat(action, self.num_factors)

        return action_unflat

    def update_belief_current_state(self, observation_unflat, action_unflat=None):
        observation = self.convert_unflat_oberservation(observation_unflat, self.multi_idx_list_observations)

        if action_unflat is None:
            prior = self.belief_current_state
        else:
            action = make_action_flat(action_unflat)
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

        efe =  utility + info_gain
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
    @staticmethod
    def convert_unflat_oberservation(observation_unflat, multi_index_list_observations):
        return multi_index_list_observations.index(tuple(observation_unflat))


def make_A_flat(A, multi_idx_list_observations):
    """ go from [p(o^1|s^1,..., s^f), ..., p(o^m|s^1,..., s^f)] to p(o|s) """
    A_flat_states = [flatten_state_factors(A_i) for A_i in A]

    A_flat = combine_observation_modalities(A_flat_states, multi_idx_list_observations)

    return A_flat


def flatten_state_factors(A_m):
    """ go from p(o^i|s^1,..., s^f) to p(o^i|s)"""  # TODO: rewrite so that it uses multi_idx_list_states
    # Get the size of the first dimension
    num_obs = A_m.shape[0]

    # Reshape the matrix
    A_flat_states_m = np.reshape(A_m, (num_obs, -1))

    return A_flat_states_m


def combine_observation_modalities(A_flat_states, multi_idx_list_observations):
    """ go from p(o^1, ..., o^m|s) to p(o|s)
    :param multi_idx_list_observations:
    """

    num_states = len(A_flat_states[0][0, :])
    num_modalities = len(A_flat_states)

    num_observations = len(multi_idx_list_observations)
    A_flat = np.zeros((num_observations, num_states))

    for idx_s in range(num_states):

        for single_idx_obs, multi_idx_obs in enumerate(multi_idx_list_observations):
            # compute list of p(o^m | s) for different modalities m and fixed state s = ss[idx_s]
            list_probs = [A_flat_states[m][multi_idx_obs[m], idx_s] for m in range(num_modalities)]

            joint_prob = product_of_elements(list_probs)

            A_flat[single_idx_obs, idx_s] = joint_prob

    return A_flat


def make_B_flat(B, multi_idx_list_states):
    """
    go from [p(s^1_tau|s^1_{tau-1}, u_{tau-1}), ..., p(s^f_tau|s^f_{tau-1}, u_{tau-1})] to
    p(s_tau|s_{tau-1}, u_{tau-1}) """
    num_factors = len(B)
    num_actions = max([B[f].shape[-1] for f in range(num_factors)])

    B = preprocess_B(B, num_actions)

    num_states = len(multi_idx_list_states)

    B_flat = np.zeros((num_states, num_states, num_actions))

    for idx_action in range(num_actions):

        for single_idx_state_tau, multi_idx_state_tau in enumerate(multi_idx_list_states):
            for single_idx_state_tau_prev, multi_idx_state_tau_prev in enumerate(multi_idx_list_states):
                list_probs = [B[f][multi_idx_state_tau[f], multi_idx_state_tau_prev[f], idx_action] for f in
                              range(num_factors)]
                joint_prob = product_of_elements(list_probs)

                B_flat[single_idx_state_tau, single_idx_state_tau_prev, idx_action] = joint_prob

    return B_flat


def preprocess_B(B, num_actions):
    """ give uncontrollable transition dynamics same number of actions """
    B_new = []
    for B_i in B:
        if B_i.shape[-1] < num_actions:
            B_i = np.repeat(B_i[:, :, 0][:, :, np.newaxis], num_actions, axis=2)
        B_new.append(B_i)
    return B_new


def make_C_flat(C, multi_idx_list_observations):
    num_observations = len(multi_idx_list_observations)

    C_flat = np.zeros(num_observations)

    for single_idx_observation, multi_idx_observation in enumerate(multi_idx_list_observations):
        for modality, observation in enumerate(multi_idx_observation):
            C_flat[single_idx_observation] += C[modality][observation]

    return C_flat


def make_D_flat(D, multi_idx_list_states):
    num_states = len(multi_idx_list_states)

    D_flat = np.zeros(num_states)

    for single_idx_state, multi_idx_state in enumerate(multi_idx_list_states):
        list_probs = [D_f[multi_idx_state[f]] for f, D_f in D]
        joint_prob = product_of_elements(list_probs)
        D_flat[single_idx_state] = joint_prob

    return D_flat

def make_action_unflat(action, num_factors, factor_idx=0):
    action_unflat = np.zeros(num_factors,dtype=int)
    action_unflat[factor_idx] = action
    return action_unflat

def make_action_flat(action_unflat, factor_idx=0):
    return action_unflat[factor_idx]


def get_multi_idx_list_states(B):
    num_states_per_factor = [B_i.shape[0] for B_i in B]
    multi_idx_list_states = list(itertools.product(*[range(s) for s in num_states_per_factor]))
    return multi_idx_list_states

def get_multi_idx_list_observations(A):
    num_observations_per_modality = [A_i.shape[0] for A_i in A]
    multi_idx_list_observations = list(itertools.product(*[range(s) for s in num_observations_per_modality]))
    return multi_idx_list_observations

def product_of_elements(lst):
    result = 1
    for num in lst:
        result *= num
    return result


