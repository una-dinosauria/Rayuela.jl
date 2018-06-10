
from __future__ import division

import os
import julia
import numpy as np

# Otherwise we leak GPU memory. See https://github.com/JuliaGPU/CuArrays.jl/releases/tag/v0.6.1
os.environ['CUARRAYS_MANAGED_POOL'] = 'false'

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
# from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import AbstractTAFunc
# from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Create a julia object
# print("Creating julia object")
# j = julia.Julia()
# print("done")

# # Get the function that we care about
# print("Importing functions")
# j.include("smac/test_lsq.jl")
# j.add_module_functions("smac_util")
# # rdqb = j.eval("run_demos_query_base")  # Load the function in Julia
# print("done")

dataset = "labelme"  # "MNIST"
m = 8
h = 256
niter = 25

j = julia.Julia()
# j.add_module_functions("Base")
j.include("smac/test_lsq.jl")
# j.add_module_functions("smac_util")
# j.eval("importall smac_util")
rdqb = j.eval("smac_util.run_demos_query_base")


def recall_from_cfg(cfg):
    """ Runs MCQ from julia based on the passed configuration

    Params
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
     scores: A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    ilsiter = int(cfg["ilsiter"])
    icmiter = int(32 / ilsiter)  # Roughly controls for runtime
    npert = int(cfg["npert"])
    randord = True if cfg["randord"] == "true" else False

    # Make sure schedule is an int and p is a float32
    sr_method = cfg["SR_method"]
    schedule = int(cfg["schedule"])
    p = float(cfg["p"])

    # Full experiment run_demos_query_base
    print(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)
    # importlib.reload(julia)
    # j = julia.Julia()
    # j.add_module_functions("Base")
    # j.include("smac/test_lsq.jl")
    # j.eval("importall smac_util")
    # rdqb = j.eval("run_demos_query_base")
    # recall_at_1 = j.run_demos_query_base(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)
    recall = rdqb(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)
    # j.gc()
    # del j
    # del sys.modules["julia"]

    return 1-recall[0]  # Minimize!


# for i in np.arange(100):
#     recall_at_1 = rdqb("labelme", 8, 256, 5, "SR_D", 8, 4, True, 4, 1, float(0.5))

# Build Configuration Space which defines all parameters and their ranges
ilsiter = UniformIntegerHyperparameter("ilsiter", 2, 16, default_value=8)
npert = UniformIntegerHyperparameter("npert", 1, m-1, default_value=4)
randord = CategoricalHyperparameter("randord", ["true", "false"], default_value="true")

sr_method = CategoricalHyperparameter("SR_method", ["SR_C", "SR_D"], default_value="SR_D")
schedule = CategoricalHyperparameter("schedule", ["1", "2", "3"], default_value="1")
p = UniformFloatHyperparameter("p", 0.01, .99, default_value=0.5)

cs = ConfigurationSpace()
cs.add_hyperparameters([ilsiter, npert, randord, sr_method, schedule, p])

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "false"
                     })

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")

thing_to_call = AbstractTAFunc(recall_from_cfg, use_pynisher=False)
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=thing_to_call)

incumbent = smac.optimize()

inc_value = recall_from_cfg(incumbent)
print("Optimized Value: %.2f" % (inc_value))
