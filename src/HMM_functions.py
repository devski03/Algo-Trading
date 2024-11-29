import numpy as np


def forward_algorithm(obs, A, B_DNN, pi):
    """
    :param obs: numpy.ndarray, sequence of observations (length T), each of dimension d
    :param A: numpy.ndarray, state transition probability matrix of size (N, N)
    :param B_DNN: (function) computes the emission probability for a given state and observation vector
                    B(i, obs_t) should return the probability of obs_t given state i
                    # usually computed using parametric forms; will be substituted for a deep neural network
    :param pi: numpy.ndarray, initial probabilities (N)
    :return: alpha: numpy.ndarray, forward probabilities matrix of size (T, N)
    """
    T, d = obs.shape
    N = len(pi)
    alpha = np.zeros((T, N))

    # Initialize forward probabilities
    for i in range(N):
        alpha[0, i] = pi[i] * B_DNN(i, obs[0])

    # Recursion step
    for t in range(1, T):
        for j in range(N):
            # Compute alpha[t, j] as the sum of previous alphas times the transition probabilities
            alpha[t, j] = sum(alpha[t - 1, i] * A[i, j] for i in range(N)) * B_DNN(j, obs[t])

    return alpha


def backward_algorithm(obs, A, B_DNN):
    """
    :param obs: numpy.ndarray, sequence of observations (length T), each of dimension d
    :param A: numpy.ndarray, state transition probability matrix of size (N, N)
    :param B_DNN: (function) computes the emission probability for a given state and observation vector
                    B(i, obs_t) should return the probability of obs_t given state i
                    # computation derived from conclusions of DNN on previous period
                    # DNN function is refined in the maximization step
    :return: beta: numpy.ndarray, backward probabilities matrix of size (T, N)
    """
    T, d = obs.shape
    N = A.shape[0]
    beta = np.zeros((T, N))

    # Initialization step (time t=T-1)
    for i in range(N):
        beta[T - 1, i] = 1  # Final beta values are initialized to 1

    # Recursion step (backwards in time)
    for t in range(T - 2, -1, -1):  # t goes from T-2 to 0
        for i in range(N):
            beta[t, i] = sum(A[i, j] * B_DNN(j, obs[t + 1]) * beta[t + 1, j] for j in range(N))

    return beta


def posterior_probabilities(alpha, beta):
    """
    Compute the posterior probabilities for each state at each time step
    :param alpha: (numpy.ndarray): Forward probabilities of shape (T, N),
                               where alpha[t, i] is the probability of the observation sequence up to time t
                               and ending in state i
    :param beta: (numpy.ndarray): Backward probabilities of shape (T, N),
                              where beta[t, i] is the probability of the observations from time t+1 to T
                              given state i at time t
    :return: gamma (numpy.ndarray): Posterior probabilities of shape (T, N),
                                   where posterior[t, i] is the probability of being in state i at time t
                                   given the full observation sequence
    """
    # Normalize to ensure probabilities sum to 1 for each time step
    gamma = (alpha * beta) / (alpha * beta).sum(axis=1, keepdims=True)

    return gamma


def expected_transitions(alpha, beta, A, B_DNN, obs):
    """
    Compute expected number of transitions between states at each time step
    :param alpha: (numpy.ndarray): Forward probabilities of shape (T, N),
                               where alpha[t, i] is the probability of the observation sequence up to time t
                               and ending in state i
    :param beta: (numpy.ndarray): Backward probabilities of shape (T, N),
                              where beta[t, i] is the probability of the observations from time t+1 to T
                              given state i at time
    :param A: (numpy.ndarray): State transition matrix of shape (N, N),
                           where A[i, j] is the probability of transitioning from state i to state j
    :param B_DNN: (function): A function that computes the emission probability
                      for a given state and observation vector. Still uses the previous prediction, to be updated.
                      B(j, obs_t) should return the probability of obs_t given state j
    :param obs: (numpy.ndarray): Sequence of observations of shape (T, d),
                             where T is the number of time steps and d is the observation dimension
    :return: xi (numpy.ndarray): Expected transitions of shape (T-1, N, N),
                            where xi[t, i, j] represents the expected number of transitions
                            from state i to state j at time t
    """
    T, d = obs.shape  # T: number of time steps, d: dimension of observations
    N = alpha.shape[1]  # Number of states

    xi = np.zeros((T - 1, N, N))

    for t in range(T - 1):
        # Compute the denominator for normalization
        denominator = 0
        for i in range(N):
            for j in range(N):
                denominator += alpha[t, i] * A[i, j] * B_DNN(j, obs[t + 1]) * beta[t + 1, j]

        # Compute xi[t, i, j]
        for i in range(N):
            for j in range(N):
                numerator = alpha[t, i] * A[i, j] * B_DNN(j, obs[t + 1]) * beta[t + 1, j]
                xi[t, i, j] = numerator / denominator

    return xi


def update_pi(gamma):
    """
    Update the initial state probabilities using the posterior probabilities
    :param gamma: (numpy.ndarray): Posterior probabilities of shape (T, N),
                                   where posterior[t, i] is the probability of being in state i
                                   at time t given the full observation sequence
    :return: pi (numpy.ndarray): Updated initial state probabilities of shape (N,),
                            where pi[i] is the probability of starting in state i
    """
    pi = gamma[0]
    return pi


def update_transitions(xi, gamma):
    """
    Updates the transition probability matrix
    :param xi: (numpy.ndarray): Expected transitions of shape (T-1, N, N),
                            where xi[t, i, j] represents the expected number of transitions
                            from state i to state j at time t
    :param gamma: (numpy.ndarray): Posterior probabilities of shape (T, N),
                               where gamma[t, i] represents the probability of being in state i at time t
    :return: A (numpy.ndarray): Updated transition probability matrix of shape (N, N),
                           where A[i, j] represents the probability of transitioning from state i to state j
    """
    T_minus_1, N, _ = xi.shape  # T-1 is the number of time steps minus 1

    # Initialize the updated transition matrix
    A = np.zeros((N, N))

    # Update A[i, j]
    for i in range(N):
        for j in range(N):
            numerator = np.sum(xi[:, i, j])  # Sum over all time steps of expected transitions from i to j
            denominator = np.sum(gamma[:-1, i])  # Sum over all time steps of being in state i
            A[i, j] = numerator / denominator if denominator != 0 else 0

    return A
