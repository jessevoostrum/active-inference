def compute_expected_free_energy_2(belief_current_state, A_trajectory, B, policy):

    belief_state_trajectories = get_belief_state_trajectories(belief_current_state, B, policy)

    belief_observation_trajectories = get_belief_observation_trajectories(belief_state_trajectories, A_trajectory)

    utility = ...

    return utility



def get_belief_state_trajectories(belief_current_state, B, policy):

    num_states = B.shape[0]

    num_steps = len(policy)

    # list of length num_states^num_steps containing tuples of length num_steps
    state_trajectories = get_state_trajectories(num_states, num_steps)

    belief_state_trajectories = np.zeros(len(state_trajectories))

    for idx, state_trajectory in enumerate(state_trajectories):

        belief_next_state = np.dot(B[state_trajectory[0], :, policy[0]], belief_current_state)

        belief_state_trajectory = belief_next_state
        for tau in range(num_steps):
            belief_state_trajectory *= B[state_trajectory[tau + 1], state_trajectory[tau], policy[tau + 1]]

        belief_state_trajectories[idx] = belief_state_trajectory

    return belief_state_trajectories


def get_state_trajectories(num_states, num_steps):
    state_trajectories = list(itertools.product(*[range(s) for s in [num_states] * num_steps]))
    return state_trajectories

def get_observation_trajectories(num_observations, num_steps):
    state_trajectories = list(itertools.product(*[range(s) for s in [num_observations] * num_steps]))
    return state_trajectories

def get_A_trajectories(A, num_steps):

    num_observations = A.shape[0]
    num_states = A.shape[1]

    state_trajectories = get_state_trajectories(num_states, num_steps)
    observation_trajectories = get_observation_trajectories(num_observations, num_steps)

    A_trajectories = np.ones((len(observation_trajectories), len(state_trajectories)))

    for idx_observation, observation_trajectory in enumerate(observation_trajectories):
        for idx_state, state_trajectory in enumerate(state_trajectories):

            for tau in range(num_steps):
                A_trajectories[idx_observation, idx_state] *= A[observation_trajectory[tau], state_trajectory[tau]]

def get_belief_observation_trajectories(belief_state_trajectories, A_trajectories):

    return np.matmul(A_trajectories, belief_state_trajectories)

