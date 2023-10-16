from evoman.environment import Environment
from evoman.controller import Controller
from controllers.demo_controller import player_controller
from controllers.genome_demo_controller import GenomeDemoController
from evolutionary_algorithm_demo import mating_tournament_selection, survivor_selection, create_offspring
import csv


def simulation(environment: Environment, controller: Controller) -> dict:
    f, p, e, t = environment.play(pcont=controller)
    return {"fitness": f, "hp_player": p, "hp_enemy": e, "game_time": t}


def save_result(statistics: dict, weights_and_biases: list, statistics_save_name: str, genome_save_name: str, group_number: int):
    directory_genome = f"neat_experiment/best_generalist_genomes/group_{group_number}/{genome_save_name}.csv"
    directory_results = f"neat_experiment/results/generalist/group_{group_number}/training_results/{statistics_save_name}.csv"
    with open(directory_genome, "w") as save_file:
        writer = csv.writer(save_file, delimiter=',')
        writer.writerow([wb for wb in weights_and_biases])

    with open(directory_results, "w+", newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["mean_fitness", "max_fitness"])
        writer.writerows(zip(statistics["mean_fitness"], statistics["max_fitness"]))


def update_statistics(statistics: dict, population: list[GenomeDemoController], generation_number: int):
    best_genome = max(population)
    average_gain = sum(genome.get_fitness() for genome in population)/len(population)
    statistics["mean_fitness"].append(average_gain)
    statistics["max_fitness"].append(best_genome.get_fitness())
    print(f"Finished Generation {generation_number}. \nBest genome this generation: {best_genome.get_fitness()}, average fitness: {average_gain}. \n")


def evaluate_genomes(genomes: list[GenomeDemoController], enemies: list):
    environment = Environment(
        logs="off",
        savelogs="no",
        multiplemode="yes",
        player_controller=player_controller(_n_hidden=10),
        enemies=enemies,
    )
    for genome in genomes:
        controller = genome.get_controller()
        result = simulation(environment=environment, controller=controller)
        gain = result['fitness']
        genome.set_fitness(fitness=gain)


def run_evolutionary_algorithm(n_generations: int, population_size: int, offspring_per_generation: int, tournament_size: int, p_recombination: float, p_mutation: float, enemies: list):
    statistics = {"mean_fitness": [], "max_fitness": []}
    population = [GenomeDemoController() for individual in range(population_size)]
    evaluate_genomes(genomes=population, enemies=enemies)
    update_statistics(statistics=statistics, population=population, generation_number=0)

    for generation in range(n_generations):
        mating_pool = mating_tournament_selection(population=population, lambda_value=offspring_per_generation, k=tournament_size)
        offspring = create_offspring(mating_pool=mating_pool, p_recombination=p_recombination, p_mutation=p_mutation)
        population += offspring
        evaluate_genomes(genomes=population, enemies=enemies)
        population = survivor_selection(population=population, mu=population_size, percentage_top_half=0.8)
        update_statistics(statistics=statistics, population=population, generation_number=generation)

    best_genome = max(population)

    return statistics, best_genome


if __name__ == "__main__":
  for group_number in [2, 1]:
        if group_number == 1:
            enemies = [1, 2, 3, 7]
        else:
            enemies = [5, 6, 8]
        n_generations = 100
        population_size = 200
        offspring_per_generation = 400
        tournament_size = 8
        p_recombination = 0.8
        p_mutation = 0.5

        n_experiments = 10

        for experiment in range(n_experiments):
            print(f"Running Experiment {experiment}")
            default_genome_name = f"demo_genome_extra_fitness_{experiment}"
            default_result_name = f"demo_result_extra_fitness_{experiment}"
            statistics, best_genome = run_evolutionary_algorithm(n_generations=n_generations, population_size=population_size, offspring_per_generation=offspring_per_generation, tournament_size=tournament_size, p_recombination=p_recombination, p_mutation=p_mutation, enemies=enemies)
            save_result(statistics=statistics, weights_and_biases=best_genome.get_weights_and_bias(), genome_save_name=default_genome_name, statistics_save_name=default_result_name, group_number=group_number)




