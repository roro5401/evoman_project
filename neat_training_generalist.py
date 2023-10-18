from evoman.environment import Environment
from evoman.controller import Controller
from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
from controllers.neat_rnn_controller import NeatRNNController
import neat
import os
import pickle
import csv

global total_generations
global enemies
global statistics


def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def evaluate_genomes(genomes: list, config: neat.Config):
    global statistics
    fitness_scores = []
    for genome_id, genome in genomes:
        controller = NeatRNNController(genome=genome, config=config)
        environment = Environment(
            logs="off",
            savelogs="no",
            multiplemode="yes",
            player_controller=controller,
            enemies=enemies,
        )
        result = simulation(environment=environment, controller=controller)
        gain = result['hp_player'] - result['hp_enemy']
        genome.fitness = gain
        fitness_scores.append(gain)

    statistics["mean_gain"].append(sum(fitness_scores) / len(fitness_scores))
    statistics["max_gain"].append(max(fitness_scores))


def run_neat(config: neat.Config, save_path: str) -> dict:
    global statistics
    statistics = {"mean_gain": [], "max_gain": []}
    population = neat.Population(config=config)
    population.add_reporter(reporter=neat.StdOutReporter(True))
    population.add_reporter(reporter=neat.StatisticsReporter())
    population.add_reporter(reporter=neat.Checkpointer(
        generation_interval=1,
        filename_prefix="neat_experiment/checkpoints/development_runs/neat-checkpoint-"
    ))

    best_genome = population.run(fitness_function=evaluate_genomes, n=total_generations)
    with open(save_path, "wb") as save_file:
        pickle.dump(best_genome, save_file)
        print(f"Succesfully saved best genome to {save_path}")
    return statistics


def run_multiple_experiments(
        config: neat.Config,
        genome_save_name: str,
        group_number: int,
        n_experiments: int,
        run_save_name: str = "run_"
):
    global enemies
    directory_name = f"neat_experiment/best_generalist_genomes/group_{group_number}/"
    results_directory = f"neat_experiment/results/generalist/group_{group_number}/training_results/"

    if group_number == 1:
        enemies = [1, 2, 3, 7]
    elif group_number == 2:
        enemies = [5, 6, 8]
    else:
        raise ValueError(f"Invalid group number. Should be 1 or 2 but you provided {group_number}.")

    for experiment in range(n_experiments):
        print(f"Running experiment no. {experiment}")
        save_path = directory_name + f"{genome_save_name}_{experiment}.pkl"
        stats = run_neat(config=config, save_path=save_path)
        with open(
                results_directory + f"{run_save_name}_{experiment}.csv",
                "w+", newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(["mean_gain", "max_gain"])
            writer.writerows(zip(stats["mean_gain"], stats["max_gain"]))


if __name__ == "__main__":
    configuration_file_name = "basic-config.txt"
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_experiment/configurations",
                               configuration_file_name)

    config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
<<<<<<< Updated upstream
    genome_save_name = "genome"
    run_save_name = "run"
=======
    genome_save_name = "genome_fitness"
    run_save_name = "run_fitness"
>>>>>>> Stashed changes
    total_generations = 100
    for group_number in [2]:
        run_multiple_experiments(
            config=config,
            group_number=group_number,
            genome_save_name=genome_save_name,
            n_experiments=10,
            run_save_name=run_save_name,
        )
