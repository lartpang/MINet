# -*- coding: utf-8 -*-
import shutil
from datetime import datetime

from config import arg_config, proj_root
from utils.misc import (
    construct_exp_name,
    construct_path,
    construct_print,
    pre_mkdir,
    set_seed,
)
from utils.solver import Solver

construct_print(f"{datetime.now()}: Initializing...")
construct_print(f"Project Root: {proj_root}")
init_start = datetime.now()

exp_name = construct_exp_name(arg_config)
path_config = construct_path(
    proj_root=proj_root,
    exp_name=exp_name,
    xlsx_name=arg_config["xlsx_name"],
)
pre_mkdir(path_config)
set_seed(seed=0, use_cudnn_benchmark=arg_config["size_list"] != None)

solver = Solver(exp_name, arg_config, path_config)
construct_print(f"Total initialization time：{datetime.now() - init_start}")

shutil.copy(f"{proj_root}/config.py", path_config["cfg_log"])
shutil.copy(f"{proj_root}/utils/solver.py", path_config["trainer_log"])

construct_print(f"{datetime.now()}: Start...")
if arg_config["resume_mode"] == "test":
    solver.test()
else:
    solver.train()
construct_print(f"{datetime.now()}: End...")
