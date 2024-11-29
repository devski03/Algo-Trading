import numpy as np
from HMM import HMM
from EmissionPredictor import EmissionPredictor
from HMM_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(obs_HMM, num_regimes, A, B, pi, obs_DNN, gamma, window_length, window_lag):
    """
    Trains hybrid HMM-DNN model using a rolling window approach
    :param window_length: length of sliding window, T
    :param window_lag: number of indices skipped when sliding window
    :param obs_HMM: numpy.ndarray (MxT1) of feature vectors for HMM model
    :param num_regimes: number of latent regimes in HMM model
    :param A: numpy.ndarray (NxN), initial transition probability matrix
    :param B: numpy.ndarray (MxN), initial emission probability matrix
    :param pi: numpy.ndarray (N) starting initial distribution vector
    :param obs_DNN: numpy.ndarray (MxT2) of feature vectors for DNN model
    :param gamma: matrix of posterior probabilities
    # Include any other relevant DNN parameters here
    :return: None
    """
    # Initialize HMM model
    hmm = HMM(obs_HMM, num_regimes, A, B, pi)

    # Set iterable parameters
    new_A = A
    new_pi = pi
    new_B = B

    # Initialize DNN model
    obs_DNN_tensor = torch.from_numpy(obs_DNN).float()
    gamma_tensor = torch.from_numpy(gamma).float()
    dataset = TensorDataset(obs_DNN_tensor, gamma_tensor)
    loader = DataLoader(dataset, batch_size=window_length, shuffle=False)
    B_dnn = EmissionPredictor(obs_DNN)

    # Run training loop
    for i in range(0, len(obs_HMM) - window_length + 1, window_lag):
        # Set training data
        training_window = obs_HMM[i:i + window_length]

        alpha = forward_algorithm(training_window, A, B, pi)
        beta = backward_algorithm(training_window, A, B)
        gamma = posterior_probabilities(alpha, beta)
        xi = expected_transitions(alpha, beta, A, B, training_window)

        new_A = update_transitions(xi, gamma)
        new_pi = update_pi(gamma)

        # Update HMM parameters A, pi
        HMM.update_transitions(new_A)
        HMM.update_transitions(new_pi)

        # Run DNN training iteration to update b_dnn function performance, and output new_B
        B_dnn.update_emissions(gamma)
        # get new_B from result of update to B_dnn, and set new_B = that
        # HMM.update_emissions(new_B)

        # if i != 0: check_convergence/stop HMM or DNN at let other finish training
        # HMM.report(), DNN.report() should provide key model fit metrics for model selection
        # e.g. weights of features over time, (A, B, pi) over time, log-likelihoods
