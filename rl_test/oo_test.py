import numpy as np
import random
import time

class Node():
    neighbors = {}
    actions = []
    agent = None
    is_terminal = False

    def __init__(self, reward):
        self.reward = reward


class Grid_world():

    def __init__(self, dim, a_cord, t_cord, base_reward, terminal_reward):
        self.n_rows = dim[0]
        self.n_cols = dim[1]
        self.create_enviroement(base_reward)
        self.link_nodes()
        self.a_row = a_cord[0]
        self.a_col = a_cord[1]
        self.enviroment[a_cord[0], a_cord[1]].agent = True
        self.a_node = self.enviroment[a_cord[0], a_cord[1]]
        self.enviroment[t_cord[0], t_cord[1]].is_terminal = True
        self.enviroment[t_cord[0], t_cord[1]].reward = terminal_reward

    def create_enviroement(self, reward):
        env = []
        for _ in range(self.n_rows):
            env_row = [Node(reward) for _ in range(self.n_cols)]
            env.append(env_row)
        self.enviroment = np.array(env)

    def link_nodes(self):
        for row, env_row in enumerate(self.enviroment):
            for col, node in enumerate(env_row):
                # adding neighbor nodes, a neighbor is non-diagonally directly connected
                actions = []
                neighbors = {}

                # handling edge cases
                # south
                if row < self.n_rows-1: 
                    actions.append('S')
                    neighbors['S'] = self.enviroment[row+1, col]

                # north
                if row > 0:
                    actions.append('N')
                    neighbors['N'] = self.enviroment[row-1, col]

                # west
                if col > 0:
                    actions.append('W')
                    neighbors['W'] = self.enviroment[row, col-1]

                # east 
                if col < self.n_cols-1:
                    actions.append('E')
                    neighbors['E'] = self.enviroment[row, col+1]

                node.actions = actions
                node.neighbors = neighbors

    def get_state(self):
        # print(f"{self.n_cols} * {self.a_row} + {self.a_col}")
        return self.n_cols * self.a_row + self.a_col

    def make_action(self, action):
        # make action
        self.a_node.agent = None
        self.a_node = self.a_node.neighbors[action]
        self.a_node.agent = True

        # update state
        a_cord = np.where(self.a_node == self.enviroment)
        self.a_row = a_cord[0][0]
        self.a_col = a_cord[1][0]

        return self.a_node.is_terminal, self.a_node.reward


    def render(self):
        boarder = '+'
        for _ in range(self.n_cols*2-1):
            boarder += '-'
        boarder += '+'
        print(boarder)

        for row, row_nodes in enumerate(self.enviroment):
            row_string = "|"
            row_string2 = " "
            for col, node in enumerate(row_nodes):
                # if (row, col) == (self.a_row, self.a_col):
                if node.agent:
                    row_string += 'A|'
                elif node.is_terminal:
                    row_string += 'T|'
                else:
                    row_string += ' |'

                row_string2 += '- '
            print(row_string)
            if row < self.n_rows-1:
                print(row_string2)

        print(boarder)


dim = (3,6)
a_cord = (2,5)
t_cord = (1,1)
base_reward = -0.1
terminal_reward = 1
grid = Grid_world(dim, a_cord, t_cord, base_reward, terminal_reward)
grid.render()

tot_reward = 0
terminal = False
while not terminal:
    time.sleep(0.2)
    actions = grid.a_node.actions
    action = random.choice(actions)
    terminal, reward = grid.make_action(action)
    tot_reward += reward
    state = grid.get_state()
    grid.render()
    print(f" state: {state}, action: {action}, total reward: {tot_reward}")













