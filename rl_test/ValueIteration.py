from MDP import GridWorld
import numpy as np
import copy

def Q_value(MDP, s, a, U):
    return sum([p * (MDP.R(s, a, s_new) + MDP.gamma * U[s_new]) for (p, s_new) in s.A[a]])
    

def value_iteration(MDP, epsilon):
    U_new = {}
    for s in MDP.S.keys():
        U_new[s] = 0

    it = 0
    while True:
        it += 1
        print(it)
        U = copy.copy(U_new)
        diff = []
        for s_ind in MDP.S.keys():
            if s_ind not in MDP.stateTerminals:
                s = MDP.S[s_ind]
                Qs = []
                for a in s.A.keys():
                    Qs.append(Q_value(MDP, s, a, U))
                U_new[s_ind] = max(Qs)
                diff.append(np.abs(U[s_ind] - U_new[s_ind]))
        MDP.plot_utility(U_new, f'value_{it}')
        if max(diff) < epsilon:
            return U


base_reward = 0.5
gamma = 0.9
epsilon = 1e-2
MDP = GridWorld(base_reward, gamma)
MDP.plot()

U = value_iteration(MDP, epsilon)
print(U)
MDP.plot_utility(U, 'value_final')


