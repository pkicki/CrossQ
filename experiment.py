import os
import time
import wandb
from wandb.integration.sb3 import WandbCallback

import jax

from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from sbx import SAC
from sbx.sac.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from sbx.sac.utils import *

import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS

from experiment_launcher import run_experiment, single_experiment

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'

experiment_time = time.time()

@single_experiment
def experiment(
    env: str = "HumanoidStandup-v4",
    algo: str = 'sac',
    seed: int = 0,
    learning_starts: int = 5000,
    log_freq: int = 300,
    wandb_entity: str = 'ias',
    wandb_project: str = 'crossQ_dmc',
    wandb_group: str = 'test_crossq',
    wandb_mode: str = 'disabled',
    eval_qbias: int = 0,
    adam_b1: float = 0.5,
    bn: bool = False,
    bn_momentum: float = 0.99,
    bn_mode: str = 'brn_actor',
    critic_activation: str = 'relu',
    crossq_style: bool = True,
    dropout: bool = False,
    ln: bool = False,
    lr: float = 1e-3,
    n_critics: int = 2,
    n_neurons: int = 256,
    policy_delay: int = 1,
    tau: float = 0.005,
    utd: int = 1,
    total_timesteps: int = 5e6,
    bnstats_live_net: bool = False,
    results_dir: str = "results/",
):

    bn = bool(bn)
    crossq_style = bool(crossq_style)
    tau = float(tau) if not crossq_style else 1.0
    bn_momentum = float(bn_momentum) if bn else 0.0
    dropout_rate, layer_norm = None, False
    policy_q_reduce_fn = jax.numpy.min
    net_arch = {'pi': [256, 256], 'qf': [n_neurons, n_neurons]}

    total_timesteps = int(total_timesteps)
    eval_freq = max(total_timesteps // log_freq, 1)

    # if 'dm_control' in env:
    #     total_timesteps = {
    #         'dm_control/reacher-easy'     : 100_000,
    #         'dm_control/reacher-hard'     : 100_000,
    #         'dm_control/ball_in_cup-catch': 200_000,
    #         'dm_control/finger-spin'      : 500_000,
    #         'dm_control/fish-swim'        : 5_000_000,
    #         'dm_control/humanoid-stand'   : 5_000_000,
    #     }[env]
    #     eval_freq = max(total_timesteps // log_freq, 1)

    td3_mode = False

    if algo == 'droq':
        dropout_rate = 0.01
        layer_norm = True
        policy_q_reduce_fn = jax.numpy.mean
        n_critics = 2
        # adam_b1 = 0.9  # adam default
        adam_b2 = 0.999  # adam default
        policy_delay = 20
        utd = 20
        group = f'DroQ_{env}_bn({bn})_ln{(ln)}_xqstyle({crossq_style}/{tau})_utd({utd}/{policy_delay})_Adam({adam_b1})_Q({net_arch["qf"][0]})'

    elif algo == 'redq':
        policy_q_reduce_fn = jax.numpy.mean
        n_critics = 10
        # adam_b1 = 0.9  # adam default
        adam_b2 = 0.999  # adam default
        policy_delay = 20
        utd = 20
        group = f'REDQ_{env}_bn({bn})_ln{(ln)}_xqstyle({crossq_style}/{tau})_utd({utd}/{policy_delay})_Adam({adam_b1})_Q({net_arch["qf"][0]})'

    elif algo == 'td3':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        td3_mode = True
        layer_norm = ln
        if dropout: 
            dropout_rate = 0.01
        group = f'TD3_{env}_bn({bn}/{bn_momentum}/{bn_mode})_ln{(ln)}_xq({crossq_style}/{tau})_utd({utd}/{policy_delay})_A{adam_b1}_Q({net_arch["qf"][0]})_l{lr}'

    elif algo == 'sac':
        # With the right hyperparameters, this here can run all the above algorithms
        # and ablations.
        layer_norm = ln
        if dropout: 
            dropout_rate = 0.01
        group = f'SAC_{env}_bn({bn}/{bn_momentum}/{bn_mode})_xq({crossq_style}/{tau})_utd({utd}/{policy_delay})_A{adam_b1}_Q({net_arch["qf"][0]})_l{lr}_st{learning_starts}_fV_.5a'

    elif algo == 'crossq':
        adam_b1 = 0.5
        policy_delay = 3
        n_critics = 2
        utd = 1                    # nice
        net_arch["qf"] = [2048, 2048]   # wider critics
        bn = True                  # use batch norm
        bn_momentum = 0.99
        crossq_style = True        # with a joint forward pass
        tau = 1.0                  # without target networks
        group = f'CrossQ_{env}'

    else:
        raise NotImplemented

    config = dict()
    config.update({
        "dropout_rate": dropout_rate,
        "layer_norm": layer_norm
    })

    with wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"seed={seed}",
        group=wandb_group,
        sync_tensorboard=True,
        config=config,
        mode=wandb_mode
    ) as wandb_run:
        
        # SLURM maintainance
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb_run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

        training_env = gym.make(env)

        if env.startswith('dm_control/'):
            for key in training_env.observation_space.spaces.keys():
                box = training_env.observation_space.spaces[key]
                if not box.shape:
                    training_env.observation_space.spaces[key] = gym.spaces.Box(box.low[None], box.high[None], (1,))
                print(key, training_env.observation_space.spaces[key])

        import optax
        model = SAC(
            "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
            training_env,
            policy_kwargs = dict({
                'activation_fn': activation_fn[critic_activation],
                'layer_norm': layer_norm,
                'batch_norm': bool(bn),
                'batch_norm_momentum': float(bn_momentum),
                'batch_norm_mode': bn_mode,
                'dropout_rate': dropout_rate,
                'n_critics': n_critics,
                'net_arch': net_arch,
                'optimizer_class': optax.adam,
                'optimizer_kwargs': dict({
                    'b1': adam_b1,
                    'b2': 0.999 # default
                })
            }),
            gradient_steps=utd,
            policy_delay=policy_delay,
            crossq_style=bool(crossq_style),
            td3_mode=td3_mode,
            use_bnstats_from_live_net=bool(bnstats_live_net),
            policy_q_reduce_fn=policy_q_reduce_fn,
            learning_starts=learning_starts,
            learning_rate=lr,
            qf_learning_rate=lr,
            tau=tau,
            gamma=0.99 if not env == 'Swimmer-v4' else 0.9999,
            verbose=0,
            buffer_size=1_000_000,
            seed=seed,
            stats_window_size=1,  # don't smooth the episode return stats over time
            tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
        )

        # Create log dir where evaluation results will be saved
        eval_log_dir = f"./eval_logs/{wandb_project}/{wandb_group + '_seed' + str(seed)}/"
        best_model_log_dir = f"./eval_logs/{wandb_project}/{wandb_group + '_seed' + str(seed)}/"
        qbias_log_dir = f"./eval_logs/{wandb_project}/{wandb_group + '_seed' + str(seed)}/"
        os.makedirs(eval_log_dir, exist_ok=True)
        if eval_qbias:
            os.makedirs(qbias_log_dir, exist_ok=True)

        # Create callback that evaluates agent
        eval_callback = EvalCallback(
            make_vec_env(env, n_envs=1, seed=seed),
            jax_random_key_for_seeds=seed,
            best_model_save_path=best_model_log_dir,
            log_path=eval_log_dir, eval_freq=eval_freq,
            n_eval_episodes=1, deterministic=True, render=False
        )

        # Callback that evaluates q bias according to the REDQ paper.
        q_bias_callback = CriticBiasCallback(
            make_vec_env(env, n_envs=1, seed=seed), 
            jax_random_key_for_seeds=seed,
            best_model_save_path=None,
            log_path=qbias_log_dir, eval_freq=eval_freq,
            n_eval_episodes=1, render=False
        )

        callback_list = CallbackList(
            [eval_callback, q_bias_callback, WandbCallback(verbose=0,)] if eval_qbias else 
            [eval_callback, WandbCallback(verbose=0,)]
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)

if __name__ == '__main__':
    run_experiment(experiment)