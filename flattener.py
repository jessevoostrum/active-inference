import itertools
import numpy as np

class Flattener:
    def __init__(self, A_unflat, B_unflat, C_unflat, D_unflat):
        self.multi_idx_list_states = self.get_multi_idx_list_states(B_unflat)
        self.multi_idx_list_observations = self.get_multi_idx_list_observations(A_unflat)

        self.A_flat = self.flatten_A(A_unflat)
        self.B_flat = self.flatten_B(B_unflat)
        self.C_flat = self.flatten_C(C_unflat)

        if D_unflat:
            self.D_flat = self.flatten_D(D_unflat)
        else:
            self.D_flat = None

        self.num_factors = len(A_unflat[0].shape) - 1

    def flatten_A(self, A):
        """ go from [p(o^1|s^1,..., s^f), ..., p(o^m|s^1,..., s^f)] to p(o|s) """
        A_flat_states = [self.flatten_state_factors(A_i) for A_i in A]

        A_flat = self.combine_observation_modalities(A_flat_states, self.multi_idx_list_observations)

        return A_flat


    def flatten_state_factors(self, A_m):
        """ go from p(o^i|s^1,..., s^f) to p(o^i|s)"""  # TODO: rewrite so that it uses multi_idx_list_states
        # Get the size of the first dimension
        num_obs = A_m.shape[0]

        # Reshape the matrix
        A_flat_states_m = np.reshape(A_m, (num_obs, -1))

        return A_flat_states_m

    def combine_observation_modalities(self, A_flat_states, multi_idx_list_observations):
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

                joint_prob = self.product_of_elements(list_probs)

                A_flat[single_idx_obs, idx_s] = joint_prob

        return A_flat


    def flatten_B(self, B):
        """
        go from [p(s^1_tau|s^1_{tau-1}, u_{tau-1}), ..., p(s^f_tau|s^f_{tau-1}, u_{tau-1})] to
        p(s_tau|s_{tau-1}, u_{tau-1}) """
        num_factors = len(B)
        num_actions = max([B[f].shape[-1] for f in range(num_factors)])

        B = self.preprocess_B(B, num_actions)

        num_states = len(self.multi_idx_list_states)

        B_flat = np.zeros((num_states, num_states, num_actions))

        for idx_action in range(num_actions):

            for single_idx_state_tau, multi_idx_state_tau in enumerate(self.multi_idx_list_states):
                for single_idx_state_tau_prev, multi_idx_state_tau_prev in enumerate(self.multi_idx_list_states):
                    list_probs = [B[f][multi_idx_state_tau[f], multi_idx_state_tau_prev[f], idx_action] for f in
                                  range(num_factors)]
                    joint_prob = self.product_of_elements(list_probs)

                    B_flat[single_idx_state_tau, single_idx_state_tau_prev, idx_action] = joint_prob

        return B_flat

    def preprocess_B(self, B, num_actions):
        """ give uncontrollable transition dynamics same number of actions """
        B_new = []
        for B_i in B:
            if B_i.shape[-1] < num_actions:
                B_i = np.repeat(B_i[:, :, 0][:, :, np.newaxis], num_actions, axis=2)
            B_new.append(B_i)
        return B_new

    def flatten_C(self, C):
        num_observations = len(self.multi_idx_list_observations)

        C_flat = np.zeros(num_observations)

        for single_idx_observation, multi_idx_observation in enumerate(self.multi_idx_list_observations):
            for modality, observation in enumerate(multi_idx_observation):
                C_flat[single_idx_observation] += C[modality][observation]

        return C_flat

    def flatten_D(self, D):
        num_states = len(self.multi_idx_list_states)

        D_flat = np.zeros(num_states)

        for single_idx_state, multi_idx_state in enumerate(self.multi_idx_list_states):
            list_probs = [D_f[multi_idx_state[f]] for f, D_f in D]
            joint_prob = self.product_of_elements(list_probs)
            D_flat[single_idx_state] = joint_prob

        return D_flat

    def unflatten_action(self, action, factor_idx=0):
        action_unflat = np.zeros(self.num_factors, dtype=int)
        action_unflat[factor_idx] = action
        return action_unflat

    def flatten_action(self, action_unflat, factor_idx=0):
        return action_unflat[factor_idx]

    def flatten_observation(self, observation_unflat):
        return self.multi_idx_list_observations.index(tuple(observation_unflat))

    def get_multi_idx_list_states(self, B):
        num_states_per_factor = [B_i.shape[0] for B_i in B]
        multi_idx_list_states = list(itertools.product(*[range(s) for s in num_states_per_factor]))
        return multi_idx_list_states
    def get_multi_idx_list_observations(self, A):
        num_observations_per_modality = [A_i.shape[0] for A_i in A]
        multi_idx_list_observations = list(itertools.product(*[range(s) for s in num_observations_per_modality]))
        return multi_idx_list_observations

    @staticmethod
    def product_of_elements(lst):
        result = 1
        for num in lst:
            result *= num
        return result