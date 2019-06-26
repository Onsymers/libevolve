# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------
# author     = "Sameh K. Mohamed"
# copyright  = "Copyright 2019, The Project"
# credits    = ["Sameh K. Mohamed"]
# license    = "MIT"
# version    = "0.0.0"
# maintainer = "Sameh K. Mohamed"
# email      = "sameh.kamaleldin@gmail.com"
# status     = "Development"
# -----------------------------------------------------------------------------------------
# Created by sameh at 2019-06-16
# -----------------------------------------------------------------------------------------

from collections.abc import Iterable
import random
import numpy as np
from deap import base, algorithms, creator, tools



# def mutate(individual, indpb, myself):
#     for i in range(len(individual)):
#         if random.random() < indpb:
#             individual[i] = myself.parameters[i].get_rand_value()
#     return individual,



class GeneticAlgorithm:
    """ A class for a generic genetic algorithm
    """
    def __init__(self, parameters, fitness_function, objective_weights):
        """ Initialise a new instance of the `GeneticAlgorithm` class

        Parameters
        ----------
        parameters : Iterable
            the set of evolutionary learning parameters
        fitness_function : function
            the fitness function. Expects named parameters that are equal or subset of the input parameters with the
            same names as specified in the input parameters. Must return an iterable.
        objective_weights : Iterable
            the assigned weights to the fitness function output objective values. Positive values denote maximisation
            objective while negative values represent minimisation objective of the corresponding objective output.
        """
        self.parameters = parameters
        self.fitness_function = fitness_function
        self.objective_weights = objective_weights
        # self.crossover= None
        # self.selection= None
        self.toolbox = None
        self.population = None
        self.stats = None



    def __eval_fun(self, individual):
        """ Customize the fitness function to be compatible for DEAP

        Parameters
        ----------
        individual : Iterable
            an individual in the population

        """

        return self.fitness_function(individual),


    def __mutate(self, individual, indpb):
        """ General Mutation for any Type of gene

        Parameters
        ----------
        individual : Iterable
            an individual in the population

        indpb : float between 0 and 1
            probabilty of mutation

        Returns
        -------
        Iterable
            The mutant Individual (Chromosome)

        """

        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = self.parameters[i].get_rand_value()

        return individual,



    def __intialize_toolbox(self, cxtype, mutprob, seltype, sel_attr_dict):
        """ Initialize The Toolbox for the GA

        Parameters
        ----------
        cxtype : Function
            function of Crossover in DEAP or Custom function (Compatible with DEAP)

        mutprob : float between 0 and 1
            probabilty of mutation

        seltype : Function
            function of Selection in DEAP or Custom function (Compatible with DEAP)

        sel_attr_dict : Dictionary
            dictionary of parameters for the selection function


        Returns
        -------
        ToolBox
            for testing

        """

        creator.create("FitnessMax", base.Fitness, weights=self.objective_weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        #Register Parameters to the GA
        for x in self.parameters:
            toolbox.register(x.name,x.get_rand_value)

        toolbox.register("individual", tools.initCycle, creator.Individual
                         ,[toolbox.__getattribute__(x.name) for x in self.parameters],n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.__eval_fun)


        # Register the Cross-Over Function
        toolbox.register("mate", cxtype)

        # Register the mutation Function
        toolbox.register("mutate", self.__mutate, indpb=mutprob)

        # Register the Selection Function
        toolbox.register("select", seltype, **sel_attr_dict)

        self.toolbox=toolbox
        return toolbox


    def __intialize_stats(self):
        """
            Just initialize Stats for the GA

        Returns
        ----------
            stats
        """

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        self.stats=stats
        return stats

    def __gen_population(self, n):
        """ initialize Population

        Parameters
        ----------
        n : integer Number
            Size of Population

        """

        self.population = self.toolbox.population(n=n)

    def intialize(self, cxtype_fun=tools.cxOnePoint, mutprob= 0.5, seltype_fun= tools.selTournament, sel_attr_dict = {'tournsize': 3} , n_pop=10):
        """ Initialize toolbox , stats and population for the GA

        Parameters
        ----------
        cxtype : Function
            function of Crossover in DEAP or Custom function (Compatible with DEAP)

        mutprob : float between 0 and 1
            probabilty of mutation

        seltype : Function
            function of Selection in DEAP or Custom function (Compatible with DEAP)

        sel_attr_dict : Dictionary
            dictionary of parameters for the selection function

        n_pop : integer Number
            Size of population



        """
        self.__intialize_toolbox(cxtype = cxtype_fun, mutprob = mutprob, seltype = seltype_fun, sel_attr_dict = sel_attr_dict)
        self.__intialize_stats()
        self.__gen_population(n=n_pop)


    def evolve(self, probab_crossing=0.5, probab_mutating = 0.2, num_generations = 10):
        """ Generate the GA and return the result

        Parameters
        ----------

        probab_crossing : float between 0 and 1
            probabilty of mutation

        probab_crossing : float between 0 and 1
            probabilty of mutation

        num_generations : integer Number
            number of Generations

        Return: Iterable
            first : Best Individual
            second : Best Value

        """

        population, log = algorithms.eaSimple(self.population, self.toolbox, cxpb=probab_crossing, stats=self.stats
                                              , mutpb=probab_mutating, ngen=num_generations)
        best_ind = tools.selBest(population, 1)[0]
        best_value=best_ind.fitness.values

        return best_ind,best_value


class Classic_GA(GeneticAlgorithm):

    def __init__(self,parameters, fitness_function, objective_weights):
        super().__init__(parameters, fitness_function, objective_weights)

    def result(self):

        super().intialize()
        return super().evolve(probab_crossing=0.5,probab_mutating=0.2,num_generations=10)



class Tour_cxTwo_GA(GeneticAlgorithm):
    def __init__(self, parameters, fitness_function, objective_weights):
        super().__init__(parameters, fitness_function, objective_weights)

    def result(self):

        super().intialize(cxtype_fun=tools.cxTwoPoint, mutprob= 0.5, seltype_fun= tools.selTournament,sel_attr_dict={"toursize": 3})
        return super().evolve(probab_crossing = 0.5,probab_mutating = 0.2,num_generations = 20)













