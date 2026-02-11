from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt

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


def collect_and_plot(data_dict, context, save_path=None):
    """
    Sammelt Daten in context['data_history'].
    Erstellt NUR dann einen Plot, wenn save_path angegeben ist.
    """
    # --- 1. Sicherstellen, dass die Historie existiert ---
    if 'data_history' not in context:
        context['data_history'] = {}
        
    # --- 2. Daten sammeln (immer, wenn data_dict nicht leer ist) ---
    # Wir iterieren über das hereinkommende Dictionary
    for key, value in data_dict.items():
        if key not in context['data_history']:
            context['data_history'][key] = [] # Neue Liste anlegen falls neuer Key
        context['data_history'][key].append(value)

    # --- 3. Plotten & Speichern (Nur wenn Pfad da ist) ---
    if save_path:
        # WICHTIG: Wir holen die Keys jetzt aus der HISTORIE, nicht aus data_dict
        # (da data_dict beim letzten Aufruf leer {} ist)
        plot_keys = list(context['data_history'].keys())
        num_plots = len(plot_keys)

        if num_plots == 0:
            print("Warnung: Keine Daten gesammelt, Plot kann nicht erstellt werden.")
            return

        print(f"Erstelle Plot für {num_plots} Metriken...")
        
        # Plot erstellen
        fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 3 * num_plots))
        
        # Falls es nur 1 Plot ist, axes in Liste umwandeln für einheitlichen Zugriff
        if num_plots == 1:
            axes = [axes]
            
        for i, key in enumerate(plot_keys):
            ax = axes[i]
            y_data = context['data_history'][key]
            x_data = range(len(y_data))
            
            ax.plot(x_data, y_data, label=key)
            ax.set_ylabel(key)
            ax.legend(loc="upper left")
            ax.grid(True)
            
        # Label für x-Achse nur ganz unten
        axes[-1].set_xlabel("Steps")
        
        # Speichern
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) # Speicher sofort freigeben
        print(f"Plot erfolgreich gespeichert: {save_path}")