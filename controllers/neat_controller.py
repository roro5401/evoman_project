from evoman.controller import Controller
import neat

class neat_controller(Controller):
	def __init__(self, genome, config):
		self.neural_network = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

	def control(self, params, cont=None):
		output = self.neural_network.activate(inputs=params)
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]


