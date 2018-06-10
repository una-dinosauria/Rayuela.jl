
import julia

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Create a julia object
print("Creating julia object")
j = julia.Julia()
print("done")

# Get the function that we care about
print("Importing functions")
j.include("smac/test_lsq.jl")
j.eval("importall smac_util")
rdqb = j.eval("run_demos_query_base")
print("done")

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
sr_method = CategoricalHyperparameter("SR_method", ["LSQ", "SR_C", "SR_D"], default_value="SR_D")

C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
randord = CategoricalHyperparameter("randord", ["true", "false"], default_value="true")

cs.add_hyperparameter([sr_method])


x = rdqb("MNIST", "LSQ", 60000)
