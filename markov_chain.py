import numpy as np
from generate_data import generate_data, NUMBER_OF_STATES
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, data, states_num=2, chain_len=10):
        self.initial_prob = np.zeros(states_num,)
        self.step_matrix = np.matrix(np.zeros((states_num, states_num)))
        self.chain_len = chain_len
        self.KNN = KNeighborsClassifier(n_neighbors=10)
        self._train(data)

    def _train(self, data):
        X, y = [], []  # for KNN
        for data_chain in data:
            for i, state in enumerate(data_chain):
                if i == 0:
                    self.initial_prob[state[0]] += 1
                    prev_state = state[0]
                    y.append(state[0])
                    X.append(state[1])
                else:
                    self.step_matrix[prev_state, state[0]] += 1
                    prev_state = state[0]
                    y.append(state[0])
                    X.append(state[1])
        self.initial_prob = self.initial_prob / np.sum(self.initial_prob)

        step_row_sums = self.step_matrix.sum(axis=1)
        self.step_matrix = self.step_matrix / step_row_sums

        self.KNN.fit(X, y)

    def decision(self, chain_data):
        pass

    def predict(self, data):
        out = []
        for chain in data:
            ch_pred = []
            for i, state in enumerate(chain):
                if i == 0:
                    pred_prob = (self.initial_prob + self.KNN.predict_proba([state[1]])[0])/2
                    pred = np.argmax(pred_prob)
                    prev_state = pred
                    ch_pred.append(pred)
                else:
                    pred_prob = (self.step_matrix[prev_state] + self.KNN.predict_proba([state[1]])[0])/2
                    pred = np.argmax(pred_prob)
                    prev_state = pred
                    ch_pred.append(pred)
            out.append(ch_pred)

        return out

    def gen_chain_prob(self):
        return self.initial_prob.T * self.step_matrix**(self.chain_len-1)


if __name__ == "__main__":

    chain_len = 10

    test = [generate_data(chain_len) for i in range(100)]


    test_data_to_plot = [[] for _ in range(NUMBER_OF_STATES)]
    
    for test_sample in test:

        for state in test_sample:

            test_data_to_plot[state[0]].append(state[1])

    test_data_to_plot = [np.array(state) for state in test_data_to_plot]
    
    plt.subplot(211)
    
    for i in range(NUMBER_OF_STATES):

        plt.plot(test_data_to_plot[i][:, 0], test_data_to_plot[i][:, 1], ".")

    plt.legend(["stan1", "stan2", "stan3"])
    plt.ylabel("cecha1")
    plt.xlabel("cecha2")
    
    accuracies = []
    accuracies_knn = []
    learning_set_sizes = [10, 100, 1000, 10000, 100000]
    
    for n in learning_set_sizes:
    
        data = [generate_data(chain_len) for i in range(n)]
        mc = MarkovChain(data, states_num=NUMBER_OF_STATES, chain_len=chain_len)
        # print("Gathered initial prob = \n {} \n".format(mc.initial_prob))
        # print("Gathered step matrix = \n {} \n".format(mc.step_matrix))

        # print("Probability for ending in states after {} steps = \n{}\n".format(chain_len, mc.gen_chain_prob()))
        
        out = mc.predict(test)
        # print(out)
        X_te = [s[1] for states in test for s in states]
        y_te = [[s[0] for s in states] for states in test]
        # print(y_te)

        res = np.equal(out, y_te)
        acc = np.sum(res) / np.size(res)
        print("Accuracy is {}".format(acc))
        
        knn_out = mc.KNN.predict(X_te)
        knn_res = np.equal(np.array(y_te).flatten(), knn_out)
        knn_acc = np.sum(knn_res) / np.size(knn_res)
        print("Accuracy for KNN is {}".format(knn_acc))
        
        already_plotted = np.zeros(NUMBER_OF_STATES)
        test_data_to_plot = [[] for _ in range(NUMBER_OF_STATES)]

        accuracies.append(acc)
        accuracies_knn.append(knn_acc)

    plt.subplot(212)

    plt.semilogx(learning_set_sizes, accuracies, ".")
    plt.semilogx(learning_set_sizes, accuracies_knn, ".")
    plt.grid(True)
    plt.legend(["acc", "acc_knn"])
    plt.ylabel("Dokładność klasyfikacji")
    plt.xlabel("Długość ciągu uczącego")
        
    plt.show()
