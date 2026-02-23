# LAMA_ITW
Into The Wild Project of LAMA, Robotics with Reeinforcement Learning




How to install:
We are using Python 3.12
To set up Venv use: 
"python3.12 -m venv .venv"
activate venv with "source ..../bin/activate"
"pip install -r requirements.txt"

after installing new packages, update requirements.txt with "pip freeze > requirements.txt"


Opening Robot XML in MuJoCo Viewer:
python -m mujoco.viewer --mjcf=assets/robot_file_name_here

Opening Optuna Dashboard:
optuna-dashboard sqlite:///optuna_tune.db

# General Infos:
We trained 3 different Robots: 2DOF,3DOF and a ContiuumsRobot
Each one has his own config, train, env file.
The ContinuumsRobot is still in development, so the files are found in its respective feature branch.
Our models are saved and differentiated by their timestamp, so for loading our models, you've got to recreate our folder structure. This is easiest done by launching(run.py, run3DOF.py) a full training cycle and terminating it after the first evalCallback. (might have to comment out WandB Init or run without tracking(3))

# General Structure:
the run(...).py runs a whole training run. Includes: loading configurations, launching training, creating vecEnvs, testing the Modell against our testset and Rendering one Example
the config(...) includes hyperparam for PPO and SAC and the factors/weights for the different rewards. (We mostly used PPO)
train(...).py launches the envs depending on the PC, calls EvalCallbacks and Callbacks for saving the bestModel and corisponding vecNorm data, also Callback for Stopping the Training after no Improvement. Also the Connection to Weights&Biases is initialized(might have to comment that out if you want to test on your own)
(...)_env.py is our custom env, connects the gym env with the mujocoEnv. The robot starts in a safe random configuration and the target is also placed at random. For each Step the rewards are calculated based on distance to target, distance to floor, action delta, joint limits etc. The episode is truncated if there is no distance improvement after set timesteps. The Max_timesteps help against RewardHacking. The distance_target is a relative vektor in between the TCP and Target, trough Matrixmulitplication the Coord Sys is shifted into the TCP, this helps against Overfitting and the Normalizatin not explodign when seeing unfamiliar Targetvectors.
render.py can render 2DOF/3DOF/Coninuumsenvs, creates a new env and renders the the result of the polcicy in Mujoco Viewer. It is possible to define Start Config of the robot and Target Position, also returns key metrics for debugging. A Grapgh of all the different rewards is returned, quite useful for rewardTuning.
testing(...).py tests the policy on a test set of ca. 500 cases, theses cases are random configs evenly spaced through the whole config/work space of the robot. Returns succes rate and other metrics, the failed runs can be rewatched if wanted.

# Results:
## 2DOF:
https://github.com/user-attachments/assets/a6fa7798-8fcc-4f1a-b95a-a6b7a500cf3e

## 3DOF:
https://github.com/user-attachments/assets/b82c641a-1b1a-471a-a547-05567a681654

## Continuums:
https://github.com/user-attachments/assets/5c74fa2f-3a0a-4862-86f0-3aa226b92965




