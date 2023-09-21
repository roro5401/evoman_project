import os
import pickle
import neat


from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
from evoman.environment import Environment
from evoman.controller import Controller


def load_genome(load_path: str) -> neat.DefaultGenome:
    with open(load_path, "rb") as genome_file:
        genome = pickle.load(genome_file)
    return genome


def one_simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def test_controller(
        enemies: list[int],
        controller: Controller,
        n_simulations: int
) -> dict:
    result = {}

    for enemy in enemies:
        print(f"Start simulating against enemy {enemy}...")
        result_enemy = {"fitness": [], "hp_player": [], "hp_enemy": [], "game_time": []}
        for simulation in range(n_simulations):
            environment = Environment(
                logs="off",
                savelogs="no",
                multiplemode="no",
                player_controller=controller,
                enemies=[enemy]
            )
            result_run = one_simulation(environment=environment, controller=controller)
            print(f"Succesfully completed run {simulation} for enemy {enemy}. Statistics:"
                  f"\n---------------------------------------------------------")
            for key, value in result_run.items():
                result_enemy[key].append(value)
                print(f"{key}: {value}\n")
        result[enemy] = result_enemy
        print(
            f"Finished simulating against enemy {enemy}. All statistics: {result[enemy]}"
        )

    return result


def test_folder_of_neat_controllers(folder_path: str, enemies: list, config_file: str,
                                    n_simulations: int = 5) -> dict:
    local_dir = os.path.dirname(__file__)
    for genome_file in os.listdir(folder_path):
        if 'no' in genome_file:
            config_file = "basic-config.txt"
            config_path = os.path.join(local_dir, "neat_experiment/configurations", config_file)
            config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
            genome = load_genome(load_path=os.path.join(folder_path, genome_file))
            neat_controller = NeatController(genome=genome, config=config)
        else:
            config_file = "basic-config-memory.txt"
            config_path = os.path.join(local_dir, "neat_experiment/configurations", config_file)
            config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
            genome = load_genome(load_path=os.path.join(folder_path, genome_file))
            neat_controller = NeatMemoryController(genome=genome, config=config)
        result = {}
        controller_result = test_controller(
            enemies=enemies, controller=neat_controller, n_simulations=n_simulations
        )
        result[genome_file] = controller_result
    return result


if __name__ == "__main__":
    result = test_folder_of_neat_controllers(
        folder_path="neat_experiment/best_specialist_genomes/enemy_4",
        enemies=[4],
        n_simulations=10,
        config_file="basic-config.txt"
    )
    print(result)
