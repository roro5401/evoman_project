from controllers.genome_demo_controller import GenomeDemoController
from controllers.demo_controller import player_controller

""""Function should return new list of sigma_stepsizes"""
def update_sigma_stepsize(sigma_stepsizes: list) -> list:
    pass


"""Function should return dict like below"""
def uncorrelated_mutation(genome: GenomeDemoController) -> dict:
    genome_info = GenomeDemoController.get_genome_information()
    weights_and_bias = genome_info["weights_and_bias"]
    sigma_stepsizes = genome_info["sigma_stepsizes"]

    new_sigma_stepsizes = update_sigma_stepsize(sigma_stepsizes=sigma_stepsizes)

    # do mutation on weights and biases

    new_weights_and_biases = []

    return {"weights_and_biases": new_weights_and_biases, "sigma_stepsizes": new_sigma_stepsizes}


def simple_arithmatic_recombination(genome_1: GenomeDemoController, genome_2: GenomeDemoController) -> (GenomeDemoController, GenomeDemoController):
    weighgts_and_bias_1 = genome_1.get_weights_and_bias()
    weighgts_and_bias_2 = genome_2.get_weights_and_bias()
    return


def mating_tournament_selection(population: list[GenomeDemoController]) -> list[GenomeDemoController]:
    return


def survivor_selection(population: list[GenomeDemoController]) -> list[GenomeDemoController]:
    return