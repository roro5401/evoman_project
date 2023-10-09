from controllers.genome_demo_controller import GenomeDemoController
from controllers.demo_controller import player_controller
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


def simple_arithmatic_recombination(genome_1: GenomeDemoController, genome_2: GenomeDemoController) -> (GenomeDemoController, GenomeDemoController):
    weighgts_and_bias_1 = genome_1.get_weights_and_bias()
    weighgts_and_bias_2 = genome_2.get_weights_and_bias()
    return


def mating_tournament_selection(population: list[GenomeDemoController]) -> list[GenomeDemoController]:
    return


def survivor_selection(population: list[GenomeDemoController]) -> list[GenomeDemoController]:
    return