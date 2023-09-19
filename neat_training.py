from evoman.environment import Environment
from evoman.controller import Controller
from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
import neat
import os
import pickle

global total_generations
total_generations = 100

def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def evaluate_genomes(genomes: list, config: neat.Config):
    for genome_id, genome in genomes:
        ## UNCOMMENT THE CONTROLLER YOU WOULD LIKE TO USE
        # controller = NeatController(genome=genome, config=config)
        controller = NeatMemoryController(genome=genome, config=config)
        environment = Environment(
            logs="off",
            savelogs="no",
            multiplemode="no",
            player_controller=controller,
            enemies=[5]
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


if __name__ == "__main__":
    # configuration_file_name = "basic-config.txt"
    configuration_file_name = "basic-config-memory.txt"
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_experiment/configurations", configuration_file_name)

    config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
    run_neat(config=config, save_path="neat_experiment/best_specialist_genomes/enemy_5/test_run_new_config.pkl")

