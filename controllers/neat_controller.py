from evoman.controller import Controller
import neat


class NeatController(Controller):
	def __init__(self, genome, config):
		self.neural_network = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

	def set(self, genome, n_inputs):
		pass

	def control(self, params, cont=None):
		output = self.neural_network.activate(inputs=params)
		return [round(action) for action in output]


