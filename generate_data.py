import numpy as np

# NUMBER_OF_STATES = 2
# initial_prob = np.array([0.4, 0.6])  # state1, state2
# step_matrix = np.array([[0.6, 0.4],
#                         [0.1, 0.9]])
#
#
# # Here variables for each state:
#
# s1_var = [(10, 1), (10, 1)]  # variables on state1
# s2_var = [(10, 1), (10, 1)]  # variables on state2
# state_variables = [s1_var, s2_var]


NUMBER_OF_STATES = 3
initial_prob = np.array([0.7, 0.2, 0.1])  # state1, state2
step_matrix = np.array([[0.6, 0.3, 0.1],
                        [0.1, 0.2, 0.7],
                        [0.3, 0.3, 0.4]
                        ])


# Here variables for each state:

s1_var = [(5, 2), (-10, 2)]  # variables on state1
s2_var = [(5, 2), (-2, 2)]  # variables on state2
s3_var = [(13, 2), (-6, 2)]
state_variables = [s1_var, s2_var, s3_var]


def step(step_matrix, curr_state):
    next_state = np.random.choice(NUMBER_OF_STATES, size=None, p=step_matrix[curr_state])
    return next_state


def generate_state_vars(variables):
    return [np.random.normal(mu, sigma) for mu, sigma in variables]


def generate_data(num):
    data = []
    curr_state = np.random.choice(NUMBER_OF_STATES, size=None, p=initial_prob)
    for _ in range(num):
        state_vars = generate_state_vars(state_variables[curr_state])
        data.append([curr_state, state_vars])
        curr_state = step(step_matrix, curr_state)

    return data


if __name__ == "__main__":
    d = generate_data(5)
    print(d)

