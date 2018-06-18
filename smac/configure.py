
from __future__ import division

import os
import julia
import numpy as np
import argparse

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import AbstractTAFunc
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Otherwise we leak GPU memory. See https://github.com/JuliaGPU/CuArrays.jl/releases/tag/v0.6.1
os.environ['CUARRAYS_MANAGED_POOL'] = 'false'

j = julia.Julia()
j.include("smac/test_lsq.jl")
rdqb = j.eval("smac_util.run_demos_query_base")
rdtqb = j.eval("smac_util.run_demos_train_query_base")


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
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    dataset = cfg["dataset"]
    m = int(cfg["m"])
    h = 256
    niter = 25

    ilsiter = int(cfg["ilsiter"])
    icmiter = int(32 / ilsiter)  # Roughly controls for runtime
    npert = int(cfg["npert"])
    randord = True if cfg["randord"] == "true" else False

    # Make sure schedule is an int and p is a float32
    sr_method = cfg["SR_method"]
    schedule = int(cfg["schedule"]) if "schedule" in cfg else int(0)
    p = float(cfg["p"]) if "p" in cfg else float(0)

    # Full experiment run_demos_query_base
    print(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)
    if dataset == 'labelme' or dataset == 'MNIST':
        recall = rdqb(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)
    else:
        recall = rdtqb(dataset, m, h, niter, sr_method, ilsiter, icmiter, randord, npert, schedule, p)

    print(recall[0])
    return 1-recall[0]  # Minimize!


def main():
    parser = argparse.ArgumentParser(description='Dump data of a log.')
    parser.add_argument('--dataset', type=str, default='labelme', help='dataset to run smac on')
    parser.add_argument('--m', type=int, default=8, help=' number of codebooks')

    args = parser.parse_args()

    # Fixed parameters
    dataset = CategoricalHyperparameter("dataset", [args.dataset], default_value=args.dataset)
    m = CategoricalHyperparameter("m", [str(args.m)], default_value=str(args.m))

    # Build Configuration Space which defines all parameters and their ranges
    ilsiter = UniformIntegerHyperparameter("ilsiter", 1, 16, default_value=8)
    npert = UniformIntegerHyperparameter("npert", 0, args.m-1, default_value=4)
    randord = CategoricalHyperparameter("randord", ["true", "false"], default_value="true")

    # SR parameters
    sr_method = CategoricalHyperparameter("SR_method", ["LSQ", "SR_C", "SR_D"], default_value="SR_D")
    schedule = CategoricalHyperparameter("schedule", ["1", "2", "3"], default_value="1")
    p = UniformFloatHyperparameter("p", 0., 1., default_value=0.5)

    # Schedule and p only make sense in SR
    use_schedule = InCondition(child=schedule, parent=sr_method, values=["SR_C", "SR_D"])
    use_p = InCondition(child=p, parent=sr_method, values=["SR_C", "SR_D"])

    cs = ConfigurationSpace()
    cs.add_hyperparameters([dataset, m, ilsiter, npert, randord, sr_method, schedule, p])
    cs.add_conditions([use_schedule, use_p])

    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": 200,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "false"
                         })

    # Optimize, using a SMAC-object
    thing_to_call = AbstractTAFunc(recall_from_cfg, use_pynisher=False)
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=thing_to_call)

    print("Optimizing!")
    incumbent = smac.optimize()
    inc_value = recall_from_cfg(incumbent)
    print("Optimized Value: %.2f" % (inc_value))


if __name__ == '__main__':
    main()

# for i in np.arange(100):
#     recall_at_1 = rdqb("labelme", 8, 256, 5, "SR_D", 8, 4, True, 4, 1, float(0.5))
