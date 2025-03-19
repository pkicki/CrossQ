from itertools import product

from experiment_launcher import Launcher, is_local
import os

os.environ["WANDB_API_KEY"] = "a9819ac569197dbd24b580d854c3041ad75efafd"

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 10
if LOCAL:
    N_EXPS_IN_PARALLEL = 2
else:
    N_EXPS_IN_PARALLEL = 100

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'standard'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = "crossq"

#env_name = "HalfCheetah-v4"
env_name = "Humanoid-v4"
#env_name = "Ant-v4"

experiment_name = f'crossq_{env_name}'

launcher = Launcher(
    exp_name=experiment_name,
    exp_file='experiment',
    n_seeds=N_SEEDS,
    start_seed=0,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=6,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)
launcher.add_experiment(
    env=env_name,
    algo="crossq",
    wandb_mode="online",
    wandb_entity = 'kicai',
    wandb_project = f'crossQ_{env_name}',
    total_timesteps=1_000_000,
)
launcher.run(LOCAL, TEST)

#noises = ["lowpass", "pink", "default"]
#noises = ["pink", "default"]

#for mu_lr, value_lr, constraint_lr in product(mu_lrs, value_lrs, constraint_lrs):
#for noise_type in noises:
#    launcher.add_experiment(
#        env_name=env_name,
#        noise_type__=noise_type,
#        debug=False,
#        #total_timesteps=20_000,
#        #debug=True,
#    )
#    launcher.run(LOCAL, TEST)
#
#cutoffs = [1.0, 3.0, 5.0]
#orders = [1, 2, 3]
#for cutoff, order in product(cutoffs, orders):
#    launcher.add_experiment(
#        env_name=env_name,
#        noise_type="lowpass",
#        cutoff__=cutoff,
#        order__=order,
#        debug=False,
#    )
#    launcher.run(LOCAL, TEST)