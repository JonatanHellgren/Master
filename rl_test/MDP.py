import numpy as np
import matplotlib.pyplot as plt

class State():
     def __init__(self, s_ind):
        self.is_terminal = False
        self.A = {}
        self.ind = s_ind

class GridWorld:

    def __init__(self, base_reward, gamma):

        self.base_reward = base_reward
        self.gamma = gamma
        self.nRows = 3
        self.nCols = 4
        self.stateObstacles = [6]
        self.load_obstacles_coords()
        self.stateTerminals = [4, 8]
        self.nCells = 12
        self.nStates = 11
        self.nActions = 4
        self.R = self.reward_function
        self._get_states()

    def reward_function(self, s, a, s_new):
        if s_new == 4:
            return 1
        elif s_new == 8:
            return -1
        else:
            return self.base_reward

    def load_obstacles_coords(self):
        self.obstacleCoords = []
        for obstacle_ind in self.stateObstacles:
            obstacle_coord = self._ind2coord(obstacle_ind)
            self.obstacleCoords.append(obstacle_coord)

    def _get_states(self):
        self.S = {}
        s_ind = 0
        for cell_ind in range(1, self.nCells+1):
            if cell_ind not in self.stateObstacles:
                (x, y) = self._ind2coord(cell_ind)
                s = State(s_ind)
                self._get_actions(s, cell_ind, x, y)
                self.S[cell_ind] = s

    def left_possible(self, x, y):
        if x > 0 and (x-1, y) not in self.obstacleCoords:
            return True
        else:
            return False

    def right_possible(self, x, y):
        if x < self.nCols-1 and (x+1, y) not in self.obstacleCoords:
            return True
        else:
            return False

    def up_possible(self, x, y):
        if y > 0 and (x, y-1) not in self.obstacleCoords:
            return True
        else:
            return False

    def down_possible(self, x, y):
        if y < self.nRows-1 and (x, y+1) not in self.obstacleCoords:
            return True
        else:
            return False


    def _get_actions(self, s, cell_ind, x, y):

        if self.left_possible(x, y):
            a = [(0.8, cell_ind-1)]

            if self.up_possible(x, y):
                a.append((0.1, cell_ind - self.nCols))
            else:
                a.append((0.1, cell_ind))

            if self.down_possible(x, y):
                a.append((0.1, cell_ind + self.nCols))
            else:
                a.append((0.1, cell_ind))

            s.A['left'] = a


        if self.right_possible(x, y):
            a = [(0.8, cell_ind+1)]

            if self.up_possible(x, y):
                a.append((0.1, cell_ind - self.nCols))
            else:
                a.append((0.1, cell_ind))

            if self.down_possible(x, y):
                a.append((0.1, cell_ind + self.nCols))
            else:
                a.append((0.1, cell_ind))

            s.A['right'] = a


        if self.up_possible(x, y):
            a = [(0.8, cell_ind - self.nCols)]

            if self.right_possible(x, y):
                a.append((0.1, cell_ind+1))
            else:
                a.append((0.1, cell_ind))

            if self.left_possible(x, y):
                a.append((0.1, cell_ind-1))
            else:
                a.append((0.1, cell_ind))

            s.A['up'] = a

        if self.down_possible(x, y):
            a = [(0.8, cell_ind + self.nCols)]

            if self.right_possible(x, y):
                a.append((0.1, cell_ind+1))
            else:
                a.append((0.1, cell_ind))

            if self.left_possible(x, y):
                a.append((0.1, cell_ind-1))
            else:
                a.append((0.1, cell_ind))

            s.A['down'] = a
    """
    def _get_actions(self, s, cell_ind, x, y):
        if x > 0 and (x-1, y) not in self.obstacleCoords:
            s.A['left'] = cell_ind - 1

        if x < self.nCols-1 and (x+1, y) not in self.obstacleCoords:
            s.A['right'] = cell_ind + 1

        if y > 0 and (x, y-1) not in self.obstacleCoords:
            s.A['up'] = cell_ind - self.nCols

        if y < self.nRows-1 and (x, y+1) not in self.obstacleCoords:
            s.A['down'] = cell_ind + self.nCols
    """


    def _plot_world(self):
        """
        Function for plotting the grid world
        """ 

        # boarder 
        coord = self._square_coord((0,self.nCols), (0,-self.nRows))
        xs, ys = zip(*coord)
        plt.plot(xs, ys, 'black')

        for ind in range(1, self.nCells+1):
            self._plot_state(ind, "0.99", "black")

        # obstacle states
        for obstacle_ind in self.stateObstacles:
            self._plot_state(obstacle_ind, "0.5", "black")

        # terminal states
        for terminal_ind in self.stateTerminals:
            self._plot_state(terminal_ind, "0.8", "black")

        # plot index
        for ind in range(1, self.nCells + 1):
            (x, y) = self._ind2coord(ind)
            plt.text(x + 0.8, -y -0.8, str(ind), fontsize=16,
                    horizontalalignment='center', verticalalignment='center')


    def _plot_state(self, i, alpha, color):
        """
        plot as single cell in the grid world
        inputs:
            param i: index of cell
            param alpha: opacity used when coloring
            param color: color used
        """
        # loading index for cell
        (x, y) = self._ind2coord(i)

        # getting coordinates for box edges
        coord = self._square_coord((x,x+1), (-y,-y-1))
        xs, ys = zip(*coord)

        # drawing box
        plt.fill(xs, ys, alpha)
        plt.plot(xs, ys, "black")

    def _square_coord(self, x, y):
        """
        finds the edge coordinates for cell, used to visulize cell
        inputs:
            param x: list of x-coordinates
            param y: list of y-coordinates
        outputs:
            list of list contating edge coordinates cell
        """
        return [[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]], [x[0], y[0]]]

    def _ind2coord(self, ind):
        ind = ind - 1
        x = ind % self.nCols
        y = ind // self.nCols
        return (x,y)

    def plot(self):
        plt.clf()
        self._plot_world()

        plt.title('MDP gridworld', size=16)
        plt.axis("equal")
        plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['0', '1', '2', '3'])
        plt.yticks(ticks=[-0.5, -1.5, -2.5], labels=['0', '1', '2'])
        plt.savefig('tmp.png')

    def plot_utility(self, U, name):
        plt.clf()
        self._plot_world()

        for cell_ind in U.keys():
            (x, y) = self._ind2coord(cell_ind)

            if cell_ind == 4:
                plt.text(x + 0.5, -y -0.5, str(1), fontsize=16,
                        horizontalalignment='center', verticalalignment='center')
            elif cell_ind == 8:
                plt.text(x + 0.5, -y -0.5, str(-1), fontsize=16,
                        horizontalalignment='center', verticalalignment='center')
            else:
                plt.text(x + 0.5, -y -0.5, str(round(U[cell_ind], 2)), fontsize=16,
                        horizontalalignment='center', verticalalignment='center')


        plt.title(f'MDP gridworld, gamma = {self.gamma}, base_reward = {self.base_reward}', size=16)
        plt.axis("equal")
        plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['0', '1', '2', '3'])
        plt.yticks(ticks=[-0.5, -1.5, -2.5], labels=['0', '1', '2'])
        plt.savefig(f'{name}.png')


    def get_nrows(self):
        return self.nRows

    def get_ncols(self):
        return self.nCols

    def get_stateobstacles(self):
        return self.stateObstacles

    def get_stateterminals(self):
        return self.stateTerminals

    def get_nstates(self):
        return self.nCells

    def get_nactions(self):
        return self.nActions




world = GridWorld(-0.1, 0.5)
world.plot()
world._get_states()
    

        
            



