from stable_baselines3.common.callbacks import BaseCallback
import os

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