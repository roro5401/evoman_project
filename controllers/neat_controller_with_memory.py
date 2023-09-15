from evoman.controller import Controller
import neat
import numpy as np


class NeatMemoryController(Controller):
	def __init__(self, genome, config, n_sensors=20):
		self.n_sensors = n_sensors
		self.neural_network = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)
		self.memory = np.zeros(self.n_sensors)

	def set(self, genome, n_inputs):
		pass

	def control(self, params, cont=None):
		inputs = np.concatenate((self.memory, params))
		output = self.neural_network.activate(inputs=inputs)
		self.memory = params
		return [round(action) for action in output]


