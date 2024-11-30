import numpy as np
from HiddenMarkovModel import HMM
from EmissionPredictor import EmissionPredictor

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training.core_training import train_hybrid_model


# Sample training code
# import obs_HMM and obs_DNN
# # Initialize DNN model and optimizer and learning rate
# obs_DNN_tensor = torch.from_numpy(obs_DNN).float()
# gamma_tensor = torch.from_numpy(gamma).float()
# ??? dataset = TensorDataset(obs_DNN_tensor, gamma_tensor)
# ??? loader = DataLoader(dataset, batch_size=window_length, shuffle=False)
# dnn = EmissionPredictor()
# set initial A,B,pi
# hmm = HMM(obs_HMM, num_regimes, A, B, pi)  # instantiate hmm model
# learning_rate = 0.0001  # adjust or set to function to change over time
# optimizer = torch.optim.Adam(dnn.parameters(), lr=learning_rate)
# dnn = EmissionPredictor()
# train_model(hmm, dnn, optimizer, 20, 2)

