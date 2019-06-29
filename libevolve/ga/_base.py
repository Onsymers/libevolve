# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------
# author     = "Sameh K. Mohamed"
# edit       = "Omar Farouk"
# copyright  = "Copyright 2019, The Project"
# credits    = ["Sameh K. Mohamed", "Omar Farouk"]
# license    = "MIT"
# version    = "0.0.0"
# maintainer = "Sameh K. Mohamed"
# email      = "sameh.kamaleldin@gmail.com", "omarfarouk.732@gmail.com"
# status     = "Development"
# -----------------------------------------------------------------------------------------
# Created by sameh at 2019-06-16
# Edited by Omar Farouk at 2019-06-26
# -----------------------------------------------------------------------------------------

from collections.abc import Iterable
from random import Random
import random
import numpy as np
from .util import GeneticHistory, normalise
from deap import base, algorithms, creator, tools



# def mutate(individual, indpb, myself):
#     for i in range(len(individual)):
#         if random.random() < indpb:
#             individual[i] = myself.parameters[i].get_rand_value()
#     return individual,



class GeneticAlgorithm:
    """ A class for a generic genetic algorithm
    """
    # def __init__(self,parameters, fitness_function, objective_weights):
    def __init__(self,
                 population_size=20,
                 nb_generations=20,
                 mutation_probability=0.3,
                 crossover_probability=0.5,
                 seed=1234,
                 verbose=0):
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
        # self.parameters = parameters
        # self.fitness_function = fitness_function
        # self.objective_weights = objective_weights
        # # self.crossover= None
        # # self.selection= None
        # self.toolbox = None
        # self.population = None
        # self.stats = None

        self.population_size = population_size
        self.nb_generations = nb_generations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.crossover_func = tools.cxOnePoint
        self.selection_func = tools.selTournament
        self.sel_attr_dict={'tournsize': 3}

        self.verbose = verbose
        self.seed = seed
        self.rs = Random(seed)
        self.parameters = None
        self.fitness_function = None
        self.objective_weights = None
        self.history = None




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



    def __intialize_toolbox(self):
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
        toolbox.register("mate", self.crossover_func)

        # Register the mutation Function
        toolbox.register("mutate", self.__mutate, indpb=self.mutation_probability)

        # Register the Selection Function
        toolbox.register("select", self.selection_func, **self.sel_attr_dict)

        #Initiatize the History
        self.history = tools.History()
        toolbox.decorate("mate", self.history.decorator)
        toolbox.decorate("mutate", self.history.decorator)

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

    def __gen_population(self):
        """
        initialize Population
        """

        self.population = self.toolbox.population(n=self.nb_generations)

    def __intialize(self):

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
        self.__intialize_toolbox()
        self.__intialize_stats()
        self.__gen_population()


    # def evolve(self, probab_crossing=0.5, probab_mutating = 0.2, num_generations = 10):
    def evolve(self, parameters, fitness_function, objective_weights, *args, **kwargs):

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

        self.parameters = parameters
        self.fitness_function = fitness_function
        self.objective_weights = objective_weights

        self.__intialize()
        population, log = algorithms.eaSimple(self.population, self.toolbox,
                                              cxpb=self.crossover_probability,
                                              stats=self.stats, mutpb=self.mutation_probability,
                                              ngen=self.nb_generations)
        best_ind = tools.selBest(population, 1)[0]
        best_value=best_ind.fitness.values

        return best_value, best_ind, self.history


class Classic_GA(GeneticAlgorithm):

    def __init__(self,
                 population_size=20,
                 nb_generations=20,
                 mutation_probability=0.3,
                 crossover_probability=0.5,
                 seed=1234,
                 verbose=0):

        super().__init__(population_size,
                         nb_generations,
                         mutation_probability,
                         crossover_probability,
                         seed,
                         verbose)




class Tour_cxTwo_GA(GeneticAlgorithm):
    def __init__(self,
                 population_size=20,
                 nb_generations=20,
                 mutation_probability=0.3,
                 crossover_probability=0.5,
                 seed=1234,
                 verbose=0):

        super().__init__(population_size,
                         nb_generations,
                         mutation_probability,
                         crossover_probability,
                         seed,
                         verbose)

        # when you want to make new type of genetic algorithm, you just make changes here
        self.crossover_func=tools.cxTwoPoint
        self.selection_func = tools.selTournament
        self.sel_attr_dict = {'tournsize': 3}

