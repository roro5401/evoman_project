from evoman.environment import Environment
from evoman.controller import Controller
from controllers.genome_demo_controller import GenomeDemoController
from evolutionary_algorithm_demo import mating_tournament_selection, \
    uncorrelated_mutation, simple_arithmetic_recombination, survivor_selection


global enemies


def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def evaluate_genomes(genomes: list[GenomeDemoController]):
    global statistics
    fitness_scores = []
    for genome in genomes:
        controller = genome.get_controller()
        environment = Environment(
            logs="off",
            savelogs="no",
            multiplemode="yes",
            player_controller=controller,
            enemies=enemies,
        )
        result = simulation(environment=environment, controller=controller)
        gain = result['hp_player'] - result['hp_enemy']
        genome. = gain
        fitness_scores.append(gain)

    statistics["mean_gain"].append(sum(fitness_scores) / len(fitness_scores))
    statistics["max_gain"].append(max(fitness_scores))
