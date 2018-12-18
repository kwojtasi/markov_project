import numpy as np


initial_prob = np.array([0.6, 0.4])  # state1, state2
step_matrix = np.array([[0.7, 0.3],
                        [0.1, 0.9]])


s1_var = [(10, 2), (5, 4)]  # variables on state1
s2_var = [(4, 1), (5, 1)]  # variables on state2
state_variables = [s1_var, s2_var]


def step(step_matrix, curr_state):
    next_state = np.random.choice(2, size=None, p=step_matrix[curr_state])
    return next_state


def generate_state_vars(variables):
    return [np.random.normal(mu, sigma) for mu, sigma in variables]


def generate_data(num):
    data = []
    curr_state = np.random.choice(2, size=None, p=initial_prob)
    for _ in range(num):
        state_vars = generate_state_vars(state_variables[curr_state])
        data.append([curr_state, state_vars])
        curr_state = step(step_matrix, curr_state)

    return data


if __name__ == "__main__":
    d = generate_data(5)
    print(d)

