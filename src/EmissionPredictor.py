import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EmissionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: represents the data input to the network
        :param hidden_dim: the heart of the DNN, responsible for learning patterns
        :param output_dim: produces the final prediction
        """
        super(EmissionPredictor, self).__init__()

        self.network = nn.Sequential(
            # Layers
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1) # Outputs log probabilities
        )

    def forward(self, x):
        return self.network(x)

    def update_params(self, new_params):
        # Update the parameters
        self.params = new_params


    def update_emissions(self, gamma):
        """
        Feeds the posterior probabilities into a deep neural network as soft labels
        and updates the parameters of B_DNN;lambda (function). B_DNN is initially parametrized
        by lambda_0, which is updated for each sample window of observations
        :param gamma: (numpy.ndarray): Posterior probabilities of shape (T, N),
                                   where gamma[t, i] represents the probability of being in state i at time t
        :return: None, updates parameters of DNN
        """
        # To be filled in


dnn = DNN(initial_params=[0.1, 0.2, 0.3])

# Training window loop
for i in range(10):
    # Example training data
    x = [1, 2, 3]  # Input
    y = dnn(x)  # Use the current version of B_DNN

    # Simulate parameter updates (e.g., from backpropagation)
    new_params = [p + 0.01 for p in dnn.params]
    dnn.update_params(new_params)
