import numpy as np
import random
import time

class Enviorment():

    def __init__(self, n_rows, n_cols, A_cord, symbols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.S = np.zeros((n_rows, n_cols))
        self.R = np.zeros((n_rows, n_cols))
        self.A_cord = A_cord
        self.symbols = symbols
        self.action_space = 4
        self.observation_space = n_cols * n_rows

    def get_actions(self):
        actions = []
        if self.A_cord[0] > 0:
            actions.append('N')
        if self.A_cord[0] < self.n_rows-1:
            actions.append('S')
        if self.A_cord[1] > 0:
            actions.append('W')
        if self.A_cord[1] < self.n_cols-1:
            actions.append('E')

        return actions

    def make_action(self, action):
        if action == 'N':
            self.A_cord = (self.A_cord[0]-1, self.A_cord[1])
        if action == 'S':
            self.A_cord = (self.A_cord[0]+1, self.A_cord[1])
        if action == 'E':
            self.A_cord = (self.A_cord[0], self.A_cord[1]+1)
        if action == 'W':
            self.A_cord = (self.A_cord[0], self.A_cord[1]-1)
        
    def get_state(self):
        return self.A_cord[0] * 3 + self.A_cord[1] + 1


    def render(self):
        self.print_boarder()
        self.print_state()
        self.print_boarder()


    def print_state(self):
        for row in range(self.n_rows):
            row_string = "|"
            row_string2 = " "
            for col in range(self.n_cols):
                if (row, col) == self.A_cord:
                    row_string += 'A|'
                else:
                    row_string += self.symbols[int(self.S[row, col])]
                    row_string += '|'

                row_string2 += '- '
            print(row_string)
            if row < self.n_rows-1:
                print(row_string2)


    def print_boarder(self):
        boarder = '+'
        for _ in range(self.n_cols*2-1):
            boarder += '-'
        boarder += '+'
        print(boarder)


symbols = {0: ' ', 1: 'A', 2: 'T'}
A_cord = (0,2)
n_row = 3
n_col = 3
env = Enviorment(n_row, n_col, A_cord, symbols)
env.render()
state = env.get_state()
print(f" state: {state}")
for it in range(2):
    time.sleep(1)
    actions = env.get_actions()
    action = random.choice(actions)
    state = env.get_state()
    env.make_action(action)
    env.render()
    print(f" state: {state}, action: {action}")








   
