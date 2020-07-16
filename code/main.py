import shutil
from datetime import datetime

from utils.config import arg_config, path_config, proj_root
from utils.misc import construct_print, pre_mkdir, set_seed
from utils.solver import Solver

construct_print(f"{datetime.now()}: Initializing...")
construct_print(f"Project Root: {proj_root}")
init_start = datetime.now()
set_seed(0)
pre_mkdir()
solver = Solver(arg_config, path_config)
construct_print(f"Total initialization time：{datetime.now() - init_start}")

shutil.copy(f"{proj_root}/utils/config.py", path_config["cfg_log"])
shutil.copy(f"{proj_root}/utils/solver.py", path_config["trainer_log"])

construct_print(f"{datetime.now()}: Start...")
if solver.only_test:
    solver.test()
else:
    solver.train()
construct_print(f"{datetime.now()}: End...")
