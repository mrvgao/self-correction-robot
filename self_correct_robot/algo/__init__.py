from self_correct_robot.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from self_correct_robot.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from self_correct_robot.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from self_correct_robot.algo.cql import CQL
from self_correct_robot.algo.iql import IQL
from self_correct_robot.algo.gl import GL, GL_VAE, ValuePlanner
from self_correct_robot.algo.hbc import HBC
from self_correct_robot.algo.iris import IRIS
from self_correct_robot.algo.td3_bc import TD3_BC
from self_correct_robot.algo.diffusion_policy import DiffusionPolicyUNet
from self_correct_robot.algo.act import ACT
