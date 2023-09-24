from evoman.environment import Environment
from evoman.controller import Controller
from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
import neat
import os
import pickle
import csv

global total_generations
global enemies
global statistics
global controller_type


def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def evaluate_genomes(genomes: list, config: neat.Config):
    global statistics, controller_type
    fitness_scores = []
    for genome_id, genome in genomes:
        if controller_type == "normal_controller":
            controller = NeatController(genome=genome, config=config)
        elif controller_type == "memory_controller":
            controller = NeatMemoryController(genome=genome, config=config)
        else:
            raise ValueError(f"Controller type must be  'normal_controller' or "
                             f"'memory_controller'. You provided {controller_type}.")
        environment = Environment(
            logs="off",
            savelogs="no",
            multiplemode="no",
            player_controller=controller,
            enemies=enemies
        )
        result = simulation(environment=environment, controller=controller)
        genome.fitness = result['fitness']
        fitness_scores.append(result['fitness'])

    statistics["mean_fitness"].append(sum(fitness_scores) / len(fitness_scores))
    statistics["max_fitness"].append(max(fitness_scores))


def run_neat(config: neat.Config, save_path: str) -> dict:
    global statistics
    statistics = {"mean_fitness": [], "max_fitness": []}
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
        controller_type: str,  # should be "memory_controller" or "normal_controller"
        genome_save_name: str,
        enemies: list[int],
        n_experiments: int,
        run_save_name: str = "run_"
):
    directory_name = f"neat_experiment/best_specialist_genomes/enemy_{enemies[0]}/{controller_type}/"
    results_directory = f"neat_experiment/results/enemy_{enemies[0]}/training_results/{controller_type}/"

    for experiment in range(n_experiments):
        print(f"Running experiment no. {experiment}")
        save_path = directory_name + f"{genome_save_name}_{experiment}.pkl"
        stats = run_neat(config=config, save_path=save_path)
        with open(
                results_directory + f"{run_save_name}_{experiment}.csv",
                "w+", newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(["mean_fitness", "max_fitness"])
            writer.writerows(zip(stats["mean_fitness"], stats["max_fitness"]))


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
    enemy_list = [4]
    for enemy in enemy_list:
        controller_type = "normal_controller"
        genome_save_name = "extra_genome"
        run_save_name = "extra_run"
        total_generations = 50
        enemies = [enemy]
        run_multiple_experiments(
            config=config,
            enemies=enemies,
            genome_save_name=genome_save_name,
            n_experiments=5,
            run_save_name=run_save_name,
            controller_type=controller_type
        )
