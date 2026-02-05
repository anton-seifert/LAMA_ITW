from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class AutoSaveVecNormalize(BaseCallback):
    """Speichert VecNormalize Statistiken automatisch beim Aufruf."""
    def __init__(self, save_path: str, filename="vec_normalize.pkl"):
        super().__init__()
        self.save_path = save_path
        self.filename = filename

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Trick 17: Holt sich automatisch das richtige Env vom Modell
        if self.model.get_vec_normalize_env():
            path = os.path.join(self.save_path, self.filename)
            self.model.get_vec_normalize_env().save(path)
        return True
    



def custom_evaluate(model, env, n_episodes=10, deterministic=True):
    """
    Führt Evaluierungs-Episoden aus und gibt detaillierte Metriken zurück.
    """
    episode_rewards = []
    successes = []
    duration = []
    distances = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        info = {} # Fallback

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            # VecEnv Unpacking (Wichtig für SB3)
            if isinstance(done, np.ndarray):
                done = done[0]
                info = info[0]
                reward = reward[0]
            
            total_reward += reward

        # Daten sammeln
        episode_rewards.append(total_reward)
        # ".get()" mit Default-Wert verhindert Crashs, falls Key fehlt
        successes.append(info.get("stayed_in_target", False))
        distances.append(info.get("total_distance"))
        duration.append(info.get("total_steps_passed_in_goal_range", 0))

    # Statistik
    mean_reward = np.mean(episode_rewards)
    success_rate = np.mean(successes)
    mean_duration_in_target = np.mean(duration)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_reward, success_rate, mean_duration_in_target, mean_distance, std_distance