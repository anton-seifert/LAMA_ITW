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
python -m mujoco.viewer --mjcf=assets/robot_file_name_here.xml