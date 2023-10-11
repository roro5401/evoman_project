from controllers.genome_demo_controller import GenomeDemoController
from controllers.demo_controller import player_controller
import random
import copy
import math
import numpy as np


""""Function should return new list of sigma_stepsizes"""
def update_sigma_stepsize(sigma_stepsizes: list) -> list:
    boundary = 0.01
    new_sigma_stepsizes = []
    n = len(sigma_stepsizes)
    tau = 1/math.sqrt(2*n)
    tau_accent = 1/math.sqrt(2*math.sqrt(n))
    N_0_1 = np.random.normal(0,1,1)
    N_0_1_list = np.random.normal(0,1,n)
    for i in range(0,n):
        new_sigma = sigma_stepsizes[i]*math.exp(tau_accent*N_0_1 + tau*N_0_1_list[i])
        if new_sigma < boundary:
            new_sigma = boundary
        new_sigma_stepsizes.append(new_sigma)
    return new_sigma_stepsizes


"""Function should return dict like below"""
def uncorrelated_mutation(genome: GenomeDemoController) -> dict:
    genome_info = GenomeDemoController.get_genome_information()
    weights_and_bias = genome_info["weights_and_bias"]
    sigma_stepsizes = genome_info["sigma_stepsizes"]

    new_sigma_stepsizes = update_sigma_stepsize(sigma_stepsizes=sigma_stepsizes)

    # do mutation on weights and biases

    new_weights_and_biases = []
    for i in range(0, len(weights_and_bias)):
        new_weight = weights_and_bias[i] + new_sigma_stepsizes[i]*np.random.normal(0,1,1)
        new_weights_and_biases.append(new_weight)

    return {"weights_and_biases": new_weights_and_biases, "sigma_stepsizes": new_sigma_stepsizes}


def simple_arithmetic_recombination(genome_1: GenomeDemoController, genome_2: GenomeDemoController) -> (GenomeDemoController, GenomeDemoController):
    genome_info1 = genome_1.get_genome_information()
    weights_and_bias1 = copy.deepcopy(genome_info1["weights_and_bias"])
    sigma_stepsizes1 = genome_info1["sigma_stepsizes"]
    genome_info2 = copy.deepcopy(genome_2.get_genome_information())
    weights_and_bias2 = genome_info2["weights_and_bias"]
    sigma_stepsizes2 = genome_info2["sigma_stepsizes"]
    k = random.randint(0,len(weights_and_bias1)-2)
    for i in range(k+1, len(weights_and_bias1)):
        avg = (weights_and_bias1[i]+weights_and_bias2[i])/2
        weights_and_bias1[i] = avg
        weights_and_bias2[i] = avg
    newgenome_1 = GenomeDemoController(weights_and_bias1, sigma_stepsizes1)
    newgenome_2 = GenomeDemoController(weights_and_bias2, sigma_stepsizes2)
    return newgenome_1, newgenome_2


def mating_tournament_selection(population: list[GenomeDemoController], lambda_value, k) -> list[GenomeDemoController]:
    mating_pool = []  # Initialize the mating pool

    current_member = 1

    while current_member <= lambda_value:
        # Pick k individuals randomly from the population
        tournament = random.sample(population, k)

        # Compare and select the best individual based on your criteria
        best_individual = max(tournament)

        # Add the best individual to the mating pool
        mating_pool.append(best_individual)

        current_member += 1

    return mating_pool


def survivor_selection(population: list[GenomeDemoController], mu: int, percentage_top_half: float) -> list[GenomeDemoController]:
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_half = population[:len(population)//2]
    worst_half = population[len(population)//2:]
    n_selected_best_half = round(mu * percentage_top_half)
    n_selected_worst_half = len(population) - n_selected_best_half
    return random.sample(population=best_half, k=n_selected_best_half) + random.sample(population=worst_half, k=n_selected_worst_half)