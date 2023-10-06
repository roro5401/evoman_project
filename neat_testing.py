import os
import pickle
import neat
import csv
from controllers.neat_controller import NeatController
from controllers.neat_controller_with_memory import NeatMemoryController
from evoman.environment import Environment
from evoman.controller import Controller


def save_result_specialist(result: dict, controller_type: str, enemy: int, csv_name: str):
    print("Saving...")
    with open(f"neat_experiment/results/enemy_{enemy}/testing_results/{controller_type}/{csv_name}.csv", "w+", newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["controller_name", "average_gain"])
        for genome, result in result.items():
            average_gain = result[enemy]
            writer.writerow([genome, average_gain])


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
    gain_per_enemy = {enemy: [] for enemy in enemies}

    for enemy in enemies:
        print(f"Start simulating against enemy {enemy}...")
        for simulation in range(n_simulations):
            environment = Environment(
                logs="off",
                savelogs="no",
                multiplemode="no",
                player_controller=controller,
                enemies=[enemy],
                visuals=True,
                speed="normal"
            )
            result_run = one_simulation(environment=environment, controller=controller)
            gain = result_run['hp_player']-result_run['hp_enemy']
            gain_per_enemy[enemy].append(gain)
            print(f"Succesfully completed run {simulation} for enemy {enemy}. Gain: {gain}")

    return {enemy: sum(gain)/len(gain) for enemy, gain in gain_per_enemy.items()}


def test_folder_of_neat_controllers(folder_path: str, enemies: list, config_file: str,
                                    controller_type: str,
                                    n_simulations: int = 5,) -> dict:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_experiment/configurations", config_file)
    config = neat.Config(genome_type=neat.DefaultGenome,
                         reproduction_type=neat.DefaultReproduction,
                         species_set_type=neat.DefaultSpeciesSet,
                         stagnation_type=neat.DefaultStagnation,
                         filename=config_path)
    result = {}
    for genome_file in os.listdir(folder_path):
        if not genome_file.endswith(".pkl"):
            continue
        print(f"\nTesting for genome {genome_file} started...\n"
              f"---------------------------------------------------------")
        genome = load_genome(load_path=os.path.join(folder_path, genome_file))
        if controller_type == "normal_controller":
            neat_controller = NeatController(genome=genome, config=config)
        elif controller_type == "memory_controller":
            neat_controller = NeatMemoryController(genome=genome, config=config)
        else:
            raise ValueError(f"Controller type must be  'normal_controller' or "
                             f"'memory_controller'. You provided {controller_type}.")

        controller_result = test_controller(
            enemies=enemies, controller=neat_controller, n_simulations=n_simulations
        )
        result[genome_file] = controller_result
    return result


if __name__ == "__main__":
    for enemy in [6]:
        for controller_type in ["memory_controller", "normal_controller"]:
            if controller_type == "normal_controller":
                config_file = "basic-config.txt"
            else:
                config_file = "basic-config-memory.txt"
            enemies=[enemy]
            result = test_folder_of_neat_controllers(
                folder_path=f"neat_experiment/best_specialist_genomes/enemy_{enemy}/{controller_type}",
                enemies=enemies,
                n_simulations=5,
                config_file=config_file,
                controller_type=controller_type
            )
            save_result_specialist(result=result, controller_type=controller_type, enemy=enemies[0], csv_name="test")
