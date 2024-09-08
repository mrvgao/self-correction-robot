from self_correct_robot.config.config import Config
from self_correct_robot.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from self_correct_robot.config.bc_config import BCConfig
from self_correct_robot.config.bcq_config import BCQConfig
from self_correct_robot.config.cql_config import CQLConfig
from self_correct_robot.config.iql_config import IQLConfig
from self_correct_robot.config.gl_config import GLConfig
from self_correct_robot.config.hbc_config import HBCConfig
from self_correct_robot.config.iris_config import IRISConfig
from self_correct_robot.config.td3_bc_config import TD3_BCConfig
from self_correct_robot.config.diffusion_policy_config import DiffusionPolicyConfig
from self_correct_robot.config.act_config import ACTConfig
