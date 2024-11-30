from src.Baum_Welch_functions import *
import torch
import torch.nn as nn


def train_hybrid_model(hmm, obs_HMM, dnn, obs_DNN_tensor, optimizer, window_length, window_lag):
    """
    Trains hybrid HMM-DNN model using a rolling window approach
    :param obs_HMM:
    :param dnn: EmissionPredictor object
    :param hmm: Hidden Markov Model (HMM) object
    :param optimizer:
    :param window_length: length of sliding window, T
    :param window_lag: number of indices skipped when sliding window
    :param obs_DNN_tensor:
    # Include any other relevant DNN parameters here
    :return: dataframe of parameters used in E step, splice parameters used in M step, and any model fit metrics
    """

    # Set iterable parameters
    new_A = hmm.A()
    new_pi = hmm.pi()
    new_B = hmm.B()

    total_dnn_loss = []  # store in return dataframe as incremental loss time series with total at end

    # Run window sliding training loop
    for i in range(0, len(obs_HMM) - window_length + 1, window_lag):

        # Set training data
        training_window = obs_HMM[i:i + window_length]
        dnn_window = obs_DNN_tensor[i]  # adjust

        # Run HMM expectation step
        alpha = forward_algorithm(training_window, new_A, new_B, new_pi)
        beta = backward_algorithm(training_window, new_A, new_B)

        # Run HMM maximization step
        gamma = posterior_probabilities(alpha, beta)
        xi = expected_transitions(alpha, beta, new_A, new_B, training_window)

        # Functions to calculate new optimal parameters
        new_A = update_transitions(xi, gamma)
        new_pi = update_pi(gamma)

        # Update HMM object parameters A, pi
        hmm.update_transitions(new_A)
        hmm.update_transitions(new_pi)

        # Run DNN training iteration using HMM outputs as inputs, and output new_B
        predictions = []  # to be set equal to blank N row matrix at start, add each output as a column indexed t
        # set to B at end of loop for next window pass into HMM

        for t in range(window_length):
            dnn.train()
            optimizer.zero_grad()
            pseudo_labels = gamma[t]  # adjust, retrieve posterior probabilities at t

            x = dnn_window[t]  # set observation vector for input

            # iteratively compile outputs vectors into matrix of emission probabilities for window O_t
            # each of said emission matrices should be retrievable in the return dataframe (T of them)
            outputs = dnn(x)  # run forward pass of DNN
            # returns softmax emission probability outputs of observation t, i.e. x

            log_outputs = torch.log(outputs + 1e-8)  # convert to log for KLDivLoss
            criterion = nn.KLDivLoss(reduction='batchmean')
            loss = criterion(log_outputs, pseudo_labels)

            loss.backward()
            optimizer.step()

            total_dnn_loss += [loss.item()]

            # Make prediction
            dnn.eval()
            with torch.no_grad():
                prediction = dnn(x)
                predictions.append(prediction)

        new_B = predictions

        # if i != 0: check_convergence/stop HMM or DNN at let other finish training
        # HMM.report(), DNN.report() should provide key model fit metrics for model selection
        # e.g. weights of features over time, (A, B, pi) over time, log-likelihoods
