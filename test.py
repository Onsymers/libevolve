from libevolve.common._base import *
from libevolve.ga._base import *

import numpy as np
import random,string


def fitness_func(v1, v2, v3):
    """ Sample objective function for evolutionary algorithm

    :param v1: int
        int value of param 1
    :param v2: int
        int value of param 2
    :param v3: character
        char value of param 3

    :return:
    float : objective value

    """
    v1_rnd = np.sum(-np.abs(v1 - np.random.randint(50, 80, [5, ])))
    v2_rnd = np.sum(-np.abs(v1 - np.random.randint(15, 54, [5, ])))

    v3 = ord(v3)
    return -(v1 + v2 - v3) ** 2 + v1_rnd + v2_rnd

def eval_func(individual):
    return fitness_func(individual[0],individual[1],individual[2])


v3_range = [c for c in string.ascii_lowercase]
v1 = EvoIntParam("v1",min_val=0,max_val=100)
v2 = EvoIntParam("v2",min_val=0,max_val=100)
v3 = EvoParam("v3",v3_range)

params = [v1,v2,v3]

# GA = GeneticAlgorithm(params,eval_func, (1.0,))
# GA.intialize()
# ind,value = GA.evolve(num_generations= 100)

c_GA = Tour_cxTwo_GA()
value, ind, hist = c_GA.evolve(parameters=params,
                         fitness_function=eval_func, objective_weights=(1,))



print(ind , value)


