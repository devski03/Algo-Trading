class HMM:
    def __init__(self, A_0, B_0, pi_0):
        """
        :param A_0: transition probability matrix
        :param B_0: emission probability matrix
        :param pi_0: initial state distribution
        """
        self.A_0 = A_0
        self.B_0 = B_0
        self.pi_0 = pi_0
