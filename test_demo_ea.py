import os
import pickle
import neat
import csv
from controllers.demo_controller import player_controller
from evoman.environment import Environment
from evoman.controller import Controller
import numpy as np


def save_result_generalist(results: dict, group_number: str, csv_name: str, n_runs: int):
    with open(f"neat_experiment/results/generalist/group_{group_number}/testing_results/{csv_name}", "w+") as result_file:
        writer = csv.writer(result_file)
        headers = ["enemy"] + [f"run_{id}" for id in range(0, n_runs)]
        writer.writerow(headers)
        for enemy, result in results.items():
            writer.writerow([f"enemy_{enemy}"] + result)


def one_simulation(environment: Environment, controller: list) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def test_controller(
        enemies: list[int],
        controller: list,
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
                player_controller=player_controller(_n_hidden=10),
                enemies=[enemy],
                # visuals="yes",
                # speed="normal"
            )
            result_run = one_simulation(environment=environment, controller=controller)
            gain = result_run['hp_player']-result_run['hp_enemy']
            gain_per_enemy[enemy].append(gain)
            print(f"Succesfully completed run {simulation} for enemy {enemy}. Gain: {gain}")

    return gain_per_enemy


def test_folder_of_demo_controllers(folder_path: str, enemies: list,
                                    n_simulations: int = 5,) -> dict:
    local_dir = os.path.dirname(__file__)
    result = {}
    for weights_and_biases_file in os.listdir(folder_path):
        with open(os.path.join(folder_path, weights_and_biases_file), "r") as file:
            reader = csv.reader(file)
            weights_and_biases = [row for idx, row in enumerate(reader) if idx == 0]
            weights_and_biases = [float(element) for element in weights_and_biases[0]]
            weights_and_biases = np.array(weights_and_biases)
        print(f"\nTesting for genome {file} started...\n"
              f"---------------------------------------------------------")

        controller_result = test_controller(
            enemies=enemies, controller=weights_and_biases, n_simulations=n_simulations
        )
        save_result_generalist(results=controller_result, n_runs=n_simulations, csv_name=weights_and_biases_file, group_number=2)
    return result


if __name__ == "__main__":
    enemies=[5, 6, 8]
    result = test_folder_of_demo_controllers(
        folder_path=f"neat_experiment/best_generalist_genomes/group_2/",
        enemies=enemies,
        n_simulations=5
    )
