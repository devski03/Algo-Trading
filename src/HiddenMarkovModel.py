class HMM:
    def __init__(self, obs_HMM, num_regimes, A, B, pi):
        """
        :param obs_HMM: array of all observation vectors model can be trained on
        :param A: active transition probability matrix
        :param B: active emission probability matrix
        :param pi: active initial state distribution
        """
        self.obs_HMM = obs_HMM
        self.num_regimes = num_regimes
        self.A = A
        self.B = B
        self.pi = pi

    def update_transitions(self, new_A):
        # Updates the transitions matrix
        self.A = new_A

    def update_pi(self, new_pi):
        # Updates the initial state vector
        self.pi = new_pi

    def update_emissions(self, new_B):
        self.B = new_B

