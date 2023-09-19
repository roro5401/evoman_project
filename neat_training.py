from evoman.environment import Environment
from evoman.controller import Controller
from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
import neat
import os
import pickle

global total_generations
global enemies
total_generations = 100
enemies = [5]

def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def evaluate_genomes(genomes: list, config: neat.Config):
    for genome_id, genome in genomes:
        ## UNCOMMENT THE CONTROLLER YOU WOULD LIKE TO USE
        controller = NeatController(genome=genome, config=config)
        # controller = NeatMemoryController(genome=genome, config=config)
        environment = Environment(
            logs="off",
            savelogs="no",
            multiplemode="no",
            player_controller=controller,
            enemies=enemies
        )
        result = simulation(environment=environment, controller=controller)
        genome.fitness = result['fitness']


def run_neat(config: neat.Config, save_path: str):
    population = neat.Population(config=config)
    population.add_reporter(reporter=neat.StdOutReporter(True))
    population.add_reporter(reporter=neat.StatisticsReporter())
    population.add_reporter(reporter=neat.Checkpointer(
        generation_interval=1,
        filename_prefix="neat_experiment/checkpoints/development_runs/neat-checkpoint-"
    ))

    best_genome = population.run(fitness_function=evaluate_genomes, n=total_generations)
    if not save_path:
        save_path = os.path.join(os.path.dirname(__file__), "best-genome.pkl")
    with open(save_path, "wb") as save_file:
        pickle.dump(best_genome, save_file)
        print(f"Succesfully saved best genome to {save_path}")


def run_multiple_experiments(config: neat.Config, genome_save_name: str, enemies: list[int], n_experiments: int):
    directory_name = f"/neat_experiment/best_specialist_genomes/enemy_{enemies[0]}"
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)

    for experiment in range(n_experiments):
        print(f"Running experiment {experiment+1}/{n_experiments}")
        save_path = os.path.join(directory_name, f"{genome_save_name}_{experiment+1}")
        run_neat(config=config, save_path=save_path)



if __name__ == "__main__":
    configuration_file_name = "basic-config.txt"
    # configuration_file_name = "basic-config-memory.txt"
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_experiment/configurations", configuration_file_name)

    config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
    run_neat(config=config, save_path="neat_experiment/best_specialist_genomes/enemy_5/test_run_new_config_no_mem.pkl")

