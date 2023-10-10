import os
import pickle
import neat
import csv
from controllers.neat_rnn_controller import NeatRNNController
from evoman.environment import Environment
from evoman.controller import Controller


def save_result_generalist(results: dict, group_number: str, csv_name: str, enemies: list):
    with open(f"neat_experiment/results/generalist/testing_results/{group_number}/{csv_name}.csv", "w+") as result_file:
        writer = csv.writer(result_file)
        headers = ["controller_name"]+ [f"enemy_{enemy}" for enemy in enemies]
        writer.writerow(headers)
        for genome, result in results.items():
            results_genome = []
            for enemy in enemies:
                results_genome.append(result[enemy])
            writer.writerow([genome] + results_genome)


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
                visuals="yes",
                speed="normal"
            )
            result_run = one_simulation(environment=environment, controller=controller)
            gain = result_run['hp_player']-result_run['hp_enemy']
            gain_per_enemy[enemy].append(gain)
            print(f"Succesfully completed run {simulation} for enemy {enemy}. Gain: {gain}")

    return {enemy: sum(gain)/len(gain) for enemy, gain in gain_per_enemy.items()}


def test_folder_of_neat_controllers(folder_path: str, enemies: list, config_file: str,
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
        print(f"\nTesting for genome {genome_file} started...\n"
              f"---------------------------------------------------------")
        genome = load_genome(load_path=os.path.join(folder_path, genome_file))
        controller = NeatRNNController(genome=genome, config=config)

        controller_result = test_controller(
            enemies=enemies, controller=controller, n_simulations=n_simulations
        )
        result[genome_file] = controller_result
    return result


if __name__ == "__main__":
    enemies=[1, 2, 3, 7]
    result = test_folder_of_neat_controllers(
        folder_path=f"neat_experiment/best_generalist_genomes/group_1/",
        enemies=enemies,
        n_simulations=5,
        config_file="basic-config.txt",
    )
    save_result_generalist(result=result, group_number=1, csv_name="test", enemies=enemies)
