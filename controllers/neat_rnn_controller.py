from evoman.controller import Controller
import neat
import numpy as np


class NeatRNNController(Controller):
    def __init__(self, genome, config, n_sensors=20):
        self.n_sensors = n_sensors
        self.neural_network = neat.nn.RecurrentNetwork.create(genome=genome,
                                                                config=config)


    def set(self, genome, n_inputs):
        pass


    def control(self, params, cont=None):
        output = self.neural_network.activate(inputs=params)
        return [round(action) for action in output]
