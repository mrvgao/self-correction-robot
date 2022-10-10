"""
Set of global variables shared across robomimic
"""
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False

# wandb entity (eg. username or team name)
WANDB_ENTITY = "add-here"

try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python robomimic/scripts/setup_macros.py"\
    )