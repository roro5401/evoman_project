from controllers.demo_controller import player_controller
import numpy as np
from typing import Optional


class GenomeDemoController():

    def __init__(self, weights_and_biases: Optional[list], sigma_stepsizes: Optional[list], n_vars = 265):
        # initialize information for genome from, weights and biases
        if weights_and_biases is None:
            self.weights_and_biases = [np.random.uniform(low=-1, high=1) for gene in range(0, n_vars)]
        else:
            self.weights_and_biases = weights_and_biases

        if sigma_stepsizes is None:
            self.sigma_stepsizes = [np.random.normal(loc=0.0, scale=1, size=1) for sigma
                                    in range(0, n_vars)]
        else:
            self.sigma_stepsizes = sigma_stepsizes


    # this function updates the controller with the new weights and then returns it
    def get_controller(self):
        controller = player_controller(_n_hidden=1)
        controller.set(controller=self.weights_and_biases, n_inputs=20)
        return controller


    def get_genome_information(self) -> dict:
        return {"weights_and_bias": self.weights_and_biases,
                "sigma_stepsizes": self.sigma_stepsizes}


    def get_weights_and_bias(self) -> list:
        return self.weights_and_biases


    def get_sigma_stepsizes(self) -> list:
        return self.sigma_stepsizes
