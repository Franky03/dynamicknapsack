import numpy as np

class KnapsackEnvironment:

    def __init__(self, n_decision_points:int, knapsack_capacity:float):
        """
        n_decision_points: number of decision points (number of items)
        knapsack_capacity: capacity of the knapsack
        """

        super(KnapsackEnvironment, self).__init__()
        self.total_items = n_decision_points
        self.alpha = knapsack_capacity # relative to the total capacity of all items

        # state 

        self.total_profit = None # total collected profit over the instance
        self.items = None # all sampled items for the instance
        self.p = None # all sampled profits for the items of the instance
        self.ct = None # current time step in (0,...,n_decision_points)
        self.current_b = None # current capacity of our knapsack
        self.current_observation = None # current state given by revealed item, its profit,
                                        # the remainining knapsack capacity and the current time point
        
    @property
    def observation(self):
        """
            Returns the state information in the form of a dictionary that contains the current knapsack capacity,
            the revealed profit of the item, the revealed item, and the current time point (in this order).
        """

        revealed_p = self.p[self.ct]
        revealed_item = self.items[self.ct]
        self.current_observation = {"b": self.current_b, "p": revealed_p, "item": revealed_item, "t": self.ct}

        return self.current_observation
    
    @property
    def done(self):
        """
        Returns True if the instance is over, else False.
        """

        if self.ct == self.total_items:# termination criterium (all items have been revealed)
            return True
        if (self.current_b < self.items).all(): # stronger termination criterium (no remaining item fits into the knapsack)
            return True
        return False
    
    def step(self, action): # this is the transition function of the environment
        """
        Transitions from one state to the next depending on the action taken and the stochastic information.

        :param action: Action (True, False) whether to take the current item.
        :return: Returns a new observation, the reward for the given decision, and a boolean if the instance is over
        """

        assert isinstance(action, bool), "Action must be a boolean value."
        b = self.current_observation["b"]
        p = self.current_observation["p"]
        item = self.current_observation["item"]

        reward = 0.0
        if action:
            assert b > item # make sure the item fits into the knapsack
            reward = p
            self.current_b -= item # update the capacity of the knapsack
            self.total_profit += reward # update the total reward collected
        self.ct += 1 # move to the next decision point

        # we return the next state observation, the reward, and whether the instance is over
        if self.done: # if decision point was last decision point only return the profit and no next state
            final_state = {"b": self.current_b, "p": 0.0, "item": 0.0, "t": self.ct}
            return final_state, reward, self.done
        # is here that i return the "Wi+1"
        return self.observation, reward, self.done # else we return the next state observation and the profit
    
    def reset(self):
        """
        Samples a new instance by sampling items and profits and constructing the knapsack accordingly.
        :return: First observation of the new instance.
        """

        self.ct = 0
        self.total_profit = 0
        self.items = np.random.poisson(lam=5.0, size=self.total_items) # sample items from uniform distribution
        
        self.items = self.items / np.sum(self.items)  # normalize the items to the total capacity
        self.items = self.items * (self.alpha/2) # scale the items to the total capacity given by alpha

        self.current_b = self.alpha # initialize knapsack to a fraction of total item capacity given by alpha

        # menor variabilidade na geração das recompensas
        self.p = self.items + 0.1 * np.random.uniform(size=self.total_items)
        self.p = self.p / np.sum(self.p)

        return self.observation # return the observation of the first state

    

        



