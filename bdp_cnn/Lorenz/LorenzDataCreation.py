
import numpy as np

from enkf_lorenz.models import Lorenz96
from enkf_lorenz.integrator import RK4Integrator
from enkf_lorenz.utilities import forward_model
from enkf_lorenz.observation.generator import observation_generator
from enkf_lorenz.assimilation.letkf import Letkf

class Lorenz(object):
    """
    Class for creating data with the lorenzmodel coded by Tobias Finn.
    The package enkf_lorenz needs to be installed for this to work.

    """

    def __init__(self,init_days_d,stepsize_h,runtime_d, nr_vars=40, forcing=8):
        """
        Initialize the model.

        :param init_days_d: float: Days before the modelstart. Will be used as initialization.
        :param stepsize_h: float: size of timesteps to be calculated in hours.
        :param runtime_d: float: duration of the modelrun in days.
        :param nr_vars: int: number of gridpoints.
        :param forcing: Forcing of the model.

        """
        self.rnd = np.random.RandomState(42)
        self.init_days = init_days_d
        self.step = np.multiply(stepsize_h, np.divide(0.05,6))
        self.step_days = self.step.copy() * 5
        self.endtime = runtime_d + self.init_days
        self.nr_vars = nr_vars
        self.forcing = forcing

        self.results = {}


    def run_model(self,boundaries=(0,0.01),label="Truth"):
        """
        Running the model.
        Results will be stored in the instance-variable "results" which is a dictionary.
        The label is the key for the run, to be later found in the results.


        :param boundaries: tuple: Within this range the start state of the model will be generated.
        :param label: str: Name of the run, to get the results by.

        """


        start_state = self.rnd.normal(boundaries[0], boundaries[1], size=(1,self.nr_vars))

        all_steps = np.arange(0,self.endtime+self.step_days,self.step_days)
        l96_truth = Lorenz96(self.forcing, self.nr_vars)
        truth_integrator = RK4Integrator(l96_truth, dt=self.step)

        ds = forward_model( all_steps=all_steps, start_point=self.init_days,start_state=start_state,
                                 integrator=truth_integrator,nr_grids=self.nr_vars)

        self.results[label] = ds





if __name__ == "__main__":
    model = Lorenz(1000,6,365)
    model.run_model(label="Test")
    print(model.results["Test"])

