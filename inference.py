import numpy as np
import itertools

def infer_states(A, B, D, obs, policy):
    """ infer the posterior over states given observations and input. The inputs matrices using the functions below. """

    num_states = A.shape[1]
    T = len(policy) + 1
    t = len(obs)

    # list all possible state trajectories
    state_trajectories = list(itertools.product(*[range(s) for s in [num_states]*T]))

    q_unnormalized = np.zeros((len(state_trajectories),))

    for idx, state_trajectory in enumerate(state_trajectories):

        prod1 = 1
        for tau in range(t):
            prod1 *= A[obs[tau], state_trajectory[tau]]

        prod2 = D[state_trajectory[0]]
        for tau in range(1,T):
            prod2 *= B[state_trajectory[tau], state_trajectory[tau-1], policy[tau-1]]

        joint = prod1 * prod2

        q_unnormalized[idx] = joint

    q = q_unnormalized / np.sum(q_unnormalized)

    return q, state_trajectories


def make_A_flat(A):
    """ go from [p(o^1|s^1,..., s^f), ..., p(o^m|s^1,..., s^f)] to p(o|s) """
    A_flat_states = [flatten_state_factors(A_i) for A_i in A]

    A_flat = combine_observation_modalities(A_flat_states)

    return A_flat

def flatten_state_factors(A_m):
    """ go from p(o^i|s^1,..., s^f) to p(o^i|s)"""
    # Get the size of the first dimension
    num_obs = A_m.shape[0]

    # Reshape the matrix
    A_flat_states_m = np.reshape(A_m, (num_obs, -1))

    return A_flat_states_m


def combine_observation_modalities(A_flat_states):
    """ go from p(o^1, ..., o^m|s) to p(o|s) """

    num_states = len(A_flat_states[0][0, :])
    num_modalities = len(A_flat_states)
    num_observations_per_modality = [len(A_flat_states_m[:, 0]) for A_flat_states_m in A_flat_states]
    num_observations = product_of_elements(num_observations_per_modality)

    A_flat = np.zeros((num_observations, num_states))

    multi_index_list_observations = list(itertools.product(*[range(s) for s in num_observations_per_modality]))
    print(multi_index_list_observations)

    for idx_s in range(num_states):

        for single_idx_obs, multi_idx_obs in enumerate(multi_index_list_observations):
            # compute list of p(o^m | s) for different modalities m and fixed state s = ss[idx_s]
            list_probs = [A_flat_states[m][multi_idx_obs[m], idx_s] for m in range(num_modalities)]

            joint_prob = product_of_elements(list_probs)

            A_flat[single_idx_obs, idx_s] = joint_prob

    return A_flat, multi_index_list_observations


def make_B_flat(B):
    """
    go from [p(s^1_tau|s^1_{tau-1}, u_{tau-1}), ..., p(s^f_tau|s^f_{tau-1}, u_{tau-1})] to
    p(s_tau|s_{tau-1}, u_{tau-1}) """
    num_factors = len(B)
    num_states_per_factor = [B[f].shape[0] for f in range(num_factors)]
    num_states = product_of_elements(num_states_per_factor)
    num_actions = max([B[f].shape[-1] for f in range(num_factors)])

    B = preprocess_B(B, num_actions)

    B_flat = np.zeros((num_states, num_states, num_actions))

    multi_index_list_states = list(itertools.product(*[range(s) for s in num_states_per_factor]))
    print(multi_index_list_states)

    for idx_action in range(num_actions):

        for single_idx_state_tau, multi_idx_state_tau in enumerate(multi_index_list_states):
            for single_idx_state_tau_prev, multi_idx_state_tau_prev in enumerate(multi_index_list_states):

                list_probs = [B[f][multi_idx_state_tau[f], multi_idx_state_tau_prev[f], idx_action] for f in
                              range(num_factors)]
                joint_prob = product_of_elements(list_probs)

                B_flat[single_idx_state_tau, single_idx_state_tau_prev, idx_action] = joint_prob

    return B_flat, multi_index_list_states


def preprocess_B(B, num_actions):
    """ to account for uncontrollable transition dynamics that have a smaller number of action """
    B_new = []
    for B_i in B:
        if B_i.shape[-1] < num_actions:
            B_i = np.repeat(B_i[:, :, 0][:, :, np.newaxis], num_actions, axis=2)
        B_new.append(B_i)
    return B_new


def product_of_elements(lst):
    result = 1
    for num in lst:
        result *= num
    return result

def make_D_flat(D):

    num_factors = len(D)
    num_states_per_factor = [D[f].shape[0] for f in range(num_factors)]
    num_states = product_of_elements(num_states_per_factor)

    D_flat = np.zeros((num_states,))

    multi_index_list_states = list(itertools.product(*[range(s) for s in num_states_per_factor]))

    for single_idx_state, multi_idx_state in enumerate(multi_index_list_states):
        list_probs = [D[f][multi_idx_state[f]] for f in range(num_factors)]
        joint_prob = product_of_elements(list_probs)
        D_flat[single_idx_state] = joint_prob

    return D_flat

