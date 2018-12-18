import numpy as np
from generate_data import generate_data


class MarkovChain:
    def __init__(self, data, stats_num=2, chain_len=10):
        self.initial_prob = np.zeros(stats_num,)
        self.step_matrix = np.matrix(np.zeros((stats_num, stats_num)))
        self.chain_len = chain_len
        self._train(data)

    def _train(self, data):
        for data_chain in data:
            for i, state in enumerate(data_chain):
                if i == 0:
                    self.initial_prob[state[0]] += 1
                    prev_state = state[0]
                else:
                    self.step_matrix[prev_state, state[0]] += 1
                    prev_state = state[0]

        self.initial_prob = self.initial_prob / np.sum(self.initial_prob)

        step_row_sums = self.step_matrix.sum(axis=1)
        self.step_matrix = self.step_matrix / step_row_sums

    def decision(self, chain_data):
        pass

    def gen_chain_prob(self):
        return self.initial_prob.T * self.step_matrix**(self.chain_len-1)


if __name__ == "__main__":
    chain_len = 10
    data = [generate_data(chain_len) for i in range(1000)]
    mc = MarkovChain(data, chain_len=chain_len)
    print("Gathered initial prob = \n {} \n".format(mc.initial_prob))
    print("Gathered step matrix = \n {} \n".format(mc.step_matrix))

    print("Probability for ending in states after{} steps = \n{}\n".format(chain_len,mc.gen_chain_prob()))
