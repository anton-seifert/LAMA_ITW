import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Deine Env direkt importieren
from envs.DOF3_env import RobotWorldEnv


def objective(trial):

    #Notwendige Config Settings
    minimal_config = {
        "robot_model_path": "assets/test_robot_3DOF.xml",
        "device": "cpu",
        "render_mode": None,


        "goal_distance": 0.03,
        "max_steps": 3000,
        "truncated_distance_steps": 100,


        "distance_reward": 10,
        "energy_reward": 0.1,
        "goal_reward": 500,
        "truncated_distance_reward": 500,
        "duration_in_target": 10,
        "in_range_reward": 10,
    }

    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Sicherheitscheck für PPO: batch_size darf nicht größer als n_steps * n_envs sein
    if batch_size > n_steps:
        raise optuna.exceptions.TrialPruned()
    
    hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
    }

    n_envs = 30

    # Train Env
    train_env = make_vec_env(
        RobotWorldEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": minimal_config}
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # 3. Modell Setup
    model = PPO(
        "MultiInputPolicy", 
        train_env, 
        **hyperparams, 
        verbose=0,
        device="cpu"
    )

    # Kürzerer Intervall für das Tuning
    model.learn(total_timesteps=100_000,progress_bar=True)

    eval_env = make_vec_env(RobotWorldEnv, n_envs=1, env_kwargs={"config": minimal_config})
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False,deterministic=True)
    
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    # Cleanup
    train_env.close()
    eval_env.close()

    return mean_reward #TODO: #Standardabweichung später mit berücksichtigen!!!!!!
    







if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) 
    #TODO: Beste Parameterspeichern, vielleicht direkt in die COnfig laden? 
    # Dahsboard benutzten? 
    
    print("Beste Parameter:", study.best_params)