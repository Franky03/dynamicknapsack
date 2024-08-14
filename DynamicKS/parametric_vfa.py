import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class LinearRegression:
    def __init__(self, knapsack_capacity, k_decision_points):
        self.knapsack_capacity = knapsack_capacity
        self.k_decision_points = k_decision_points

        self.all_x = deque(maxlen=300)
        self.all_y = deque(maxlen=300)

        # parameters of linear regression
        self.a = np.zeros(k_decision_points)
        self.b = 0.0

        self.errors = []  # Armazena os erros

        # information required for online updates of linear regression
        self.n = 0
        self.meanX = np.zeros(k_decision_points)
        self.meanY = 0.0
        self.varX = np.zeros(k_decision_points)
        self.covXY = np.zeros(k_decision_points)

    def act(self, observation):
        b = observation["b"]
        item = observation["item"]
        p = observation["p"]
        ct = observation["t"]

        if b < item:
            return False
        
        remaining_b_taking = (b-item) / self.knapsack_capacity
        pds_value_taking = self.a[ct] * remaining_b_taking + self.b # ax + b

        remaining_b_not_taking = b / self.knapsack_capacity
        pds_value_not_taking = self.a[ct] * remaining_b_not_taking + self.b

        if p+pds_value_taking > pds_value_not_taking:
            return True
        return False
    
    def train(self, env, n_iterations, evaluate_every_n):

        for episode in range(n_iterations):
            current_exploration_rate = 1.0 * np.exp(-1 * episode / (n_iterations/4))
            observed_pds = []
            collected_rewards = []
            observation = env.reset()

            while True:
                if np.random.random() < current_exploration_rate:
                    if observation["b"] < observation["item"]:
                        action = np.random.random() < 0.5 # take or not
                    else:
                        action = False
                else: # exploitation
                    action = self.act(observation) # choose an action according to the current policy
                
                observation, reward, done = env.step(action)# transition to the next state
                observed_pds.append((observation["t"] - 1, observation["b"]))
                collected_rewards.append(reward)
                if done:
                    self._update(observed_pds, collected_rewards)

                    if episode % evaluate_every_n == 0:
                        current_performance = self._evaluate(env, 100)
                        mean_error = np.mean(self.errors)
                        print("Episode: {}\t Mean Profit {:.2f}\t "
                          "Exploration Rate: {:.2f}\t Mean Error: {:.4f}".format(
                              episode, current_performance, current_exploration_rate, mean_error))
                        print("Length of x: ", len(self.all_x))
                    break
    def _update(self, observed_pds, collected_rewards):
        collected_rewards.pop(0)  # delete the first reward as it does not belong to a pds
        collected_rewards.append(0.0)  # append a reward of 0.0 for the last pds
        pds_values = np.cumsum(np.array(collected_rewards)[::-1])[::-1]  # calculate the value of each pds

        for pds, value in zip(observed_pds, pds_values):
            x = np.zeros(self.k_decision_points + 1)
            x[-1] = 1 
            x[pds[0]] = pds[1] / self.knapsack_capacity # normalize the remaining capacity
            y = value
            y_hat = self.a[pds[0]] * x[pds[0]] + self.b
            grad = -2 * x * (y - y_hat)
            self.a -= 0.01 * grad[:-1]
            self.b -= 0.01 * grad[-1]

            self.all_x.append((pds[0], x[pds[0]]))
            self.all_y.append(y)

            self.errors.append(abs(y - y_hat))

    def plot_regression(self):
        all_x = [x[1] for x in self.all_x]
        all_y = self.all_y

        plt.scatter(all_x, all_y, label="Dados Reais", alpha=0.6)

        # Ajuste uma regressão linear diretamente aos dados observados
        coeffs = np.polyfit(all_x, all_y, 1)  # Ajuste linear (1º grau)
        x_line = np.linspace(min(all_x), max(all_x), 100)
        y_line = coeffs[0] * x_line + coeffs[1]  # y = ax + b com os coeficientes ajustados
        plt.plot(x_line, y_line, color='red', label="Ajuste Linear")

        plt.xlabel("Capacidade Restante Normalizada (x)")
        plt.ylabel("Recompensa Esperada (y)")
        plt.title("Ajuste de Regressão Linear")
        plt.legend()
        plt.savefig("linear_regression.png")
        

    def _evaluate(self, env, n_iterations):
        scores = [] 
        for instance_id in range(0, n_iterations):  # test a policy on 100 knapsack instances
            observation = env.reset()  # initialize a new instance and return first observation (state 0)
            while True:  # as long as the instance has not terminated
                action = self.act(observation)  # choose an action according to the policy
                observation, reward, done = env.step(action)  # transition to the next state
                if done:  # if instance is over
                    scores.append(env.total_profit)  # track achieved total profit
                    break
        return np.mean(scores)


