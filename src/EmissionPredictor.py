import torch
import torch.nn as nn


class EmissionPredictor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, conv_out_channels, dense_units, output_dim,
                 num_lstm_layers=2, num_conv_layers=2):
        """
        DNN layers:
        :param input_dim: length of input vectors at each timestep; number of states N
        :param lstm_hidden_dim:
        :param conv_out_channels:
        :param dense_units:
        :param output_dim:
        :param num_lstm_layers:
        :param num_conv_layers:
        """
        super(EmissionPredictor, self).__init__()

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Stacked 1D Convolutional layers with kernel size 3
        conv_layers = []
        in_channels = lstm_hidden_dim * 2  # Bidirectional output
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, conv_out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(conv_out_channels))
            in_channels = conv_out_channels
        self.conv = nn.Sequential(*conv_layers)

        # Dense layers with ReLU activation, gradually reducing units
        dense_layers = []
        in_features = conv_out_channels
        for units in dense_units:
            dense_layers.append(nn.Linear(in_features, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(0.3))
            in_features = units
        self.dense = nn.Sequential(*dense_layers)

        # Softmax output layer
        self.output = nn.Linear(dense_units[-1], output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM expects input in (batch, seq_len, input_dim) format
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim * 2)

        # Transpose to (batch, hidden_dim * 2, seq_len) for Conv1d
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.conv(lstm_out)

        # Global Average Pooling to reduce sequence length
        gap_out = torch.mean(conv_out, dim=2)

        # Dense layers
        dense_out = self.dense(gap_out)

        # Output layer with softmax
        output = self.output(dense_out)
        return self.softmax(output)

    # class EmissionPredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """
#         :param input_dim: represents the data input to the network
#         :param hidden_dim: the heart of the DNN, responsible for learning patterns
#         :param output_dim: produces the final prediction
#         """
#         super(EmissionPredictor, self).__init__()
#
#         self.network = nn.Sequential(
#             # Layers
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.LogSoftmax(dim=1) # Outputs log probabilities
#         )
#
#     def forward(self, x):
#         return self.network(x)


dnn = EmissionPredictor(initial_params=[0.1, 0.2, 0.3])

# Training window loop
for i in range(10):
    # Example training data
    x = [1, 2, 3]  # Input
    y = dnn(x)  # Use the current version of B_DNN

    # Simulate parameter updates (e.g., from backpropagation)
    new_params = [p + 0.01 for p in dnn.params]
    dnn.update_params(new_params)
