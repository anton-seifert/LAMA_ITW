import optuna
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO,SAC,TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils import custom_evaluate
from configurations.config_3DOF import Settings as config_dict


from envs.DOF3_env import RobotWorldEnv

#Anpassen der learning rate mit zunehmenden Fortschritt
def linear_schedule(initial_value):
    def f(progress_remaining):
        return progress_remaining * initial_value
    return f


def objective(trial):

    n_envs = 12

    # Benötigte Config damit der Spaß funktioniert
    minimal_config = {
        "robot_model_path": "assets/test_robot_3DOF.xml",
        "device": "cpu",
        "render_mode": None,
        "goal_distance": 0.03,
        "max_steps": 3000,
        "truncated_distance_steps": 100,
        "distance_reward": trial.suggest_int("distance_reward",1.0, 50.0),
        "energy_reward": trial.suggest_int("energy_reward", 0.0, 50),
        "goal_reward": trial.suggest_int("goal_reward", 100, 500),
        "truncated_distance_reward": trial.suggest_int("truncated_distance_reward", 100, 500),
        "duration_in_target": trial.suggest_int("duration_in_target",10,100),#trial.suggest_float("duration_in_target",1,50), #der hier ist mir sehr sus, das ist doch von uns gegeben
        "in_range_reward": trial.suggest_int("in_range_reward",10,100),
        "joint_limit_reward" :trial.suggest_int("joint_limit_reward",1,50),#2,
        "singularity_reward_factor" : 0, #actually punishes velocity, velocity is infinitly high in singularities
        "crash_reward": trial.suggest_int("crash_reward",10,500),
        "floor_distance_reward" : trial.suggest_int("floor_distance_reward",1,50),#10,
    }
    """
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    # Sicherheitsprüfungen (sonst crasht SB3)
    if batch_size > n_steps * n_envs:
        raise optuna.exceptions.TrialPruned()
    if (n_steps * n_envs) % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)

    # verschiedene Hyperparameter
    hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": linear_schedule(lr),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-3, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.15, 0.3),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "use_sde": True,
        "sde_sample_freq": trial.suggest_int("sde_sample_freq", 4, 16),
        "target_kl": trial.suggest_float("target_kl", 0.01, 0.1),
    }

    # verschieden Algos
    #algo = trial.suggest_categorical("algo", ["PPO", "SAC", "TD3"])

    # Architektur: PPO trennt policy (pi) und value function (vf) gerne
    net_width = trial.suggest_categorical("net_width", [64, 128, 256])
    depth = trial.suggest_int("net_depth", 1, 3)
    
    layers = [net_width] * depth
    policy_kwargs = {
        "net_arch": dict(pi=layers, vf=layers), # Explizite Trennung ist sauberer
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    }
    """
    n_steps = 2048
    batch_size = 64
    
    hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 3e-4, # Standard PPO Rate
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,       # Erstmal ohne Neugier-Bonus erzwingen
        "clip_range": 0.2,
        "n_epochs": 10,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "use_sde": True,       # Für Robotik lassen wir das an
    }

    # Standard Architektur
    policy_kwargs = {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    }

    train_env = make_vec_env(
        RobotWorldEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": minimal_config}
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        **hyperparams,
        verbose=0,
        device="cpu"
    )


    eval_env = make_vec_env(
        RobotWorldEnv,
        n_envs=4,
        env_kwargs={"config": minimal_config}
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False
    )
    # Normalisieren die Beobachtungen, Evaluierung benutzt nun gleiche Werte wie Training
    eval_env.obs_rms = train_env.obs_rms

    total_timesteps = 500_000
    eval_interval = 50_000
    n_evals = total_timesteps // eval_interval


    for i in range(n_evals):
        # Tranieren in kleineren Blöcken
        model.learn(total_timesteps=eval_interval)

        mean_reward, success_rate, mean_duration_in_target, mean_distance, std_distance= custom_evaluate(
            model,
            eval_env,
            n_episodes=5,
            deterministic= True
        )
        # Kombinierter Wert aus Mean und Std, für Stabiltät, Faktor kann/sollte angepasst werden 
        #stabiltiy_score = mean_reward - 0.15 * std_reward
        
        #calculate different scores based on their importance
        success_score = success_rate*1000 #score between 0 and 1000
        duration_score = (mean_duration_in_target/minimal_config["duration_in_target"])*100 #score between 0 and 100
        distance_score = 10*np.exp(-(mean_distance +0.25*std_distance)) #score between 0 and 10
        
        score = success_score + duration_score + distance_score
        # Score an Optuna melden
        trial.report(score, i)

        # Abrechen von schlechten Trials
        if trial.should_prune():
            train_env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

    train_env.close()
    eval_env.close()

    return score

    

if __name__ == "__main__":

     
    #TODO: Beste Parameterspeichern, vielleicht direkt in die COnfig laden? 
    # Dahsboard benutzten? 

    # Sampler: entscheidet, welche Parameter ausprobiert werden
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        seed=42,
        warn_independent_sampling=False
    )

    # Pruner: bricht schlechte Trainings frühzeitig ab
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
        interval_steps=1
    )
    db = "sqlite:///optuna_tune.db"

    study = optuna.create_study(direction="maximize",sampler=sampler,pruner=pruner,storage=db,study_name=f"PPO_3DOF_Tunen_{config_dict.get('timestamp')}",load_if_exists=True)
    study.optimize(objective, n_trials=200,show_progress_bar=True)

    print("Beste Parameter:", study.best_params)