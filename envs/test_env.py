import gymnasium as gym
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv


class RobotWorldEnv(MujocoEnv):
    
    def __init__(self, model_path: str):
    
        #load model from Path
        self.robot_model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.robot_model)

        self.target_pos = np.array(1,1,1) # just for initialition
        self.info = {}  #just for initaliation, later on gets filled with for tracking succes

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations

        # TCP Pos, TCP Acceleration, Joint Angles, Joint Velocity, Joint Acceleration, Target Pos

        """
        nv = degrees of freedom, also eig die number of velocities, or acceleration
        nq = number of generalized coords, die angle position (das kracht beim continums robot, weil dann quateriums oder so benutzt werden, nicht 3 komponenten, sondern 4)
        sensor 3 TCP Acc, 3 TCP Gyro
        kinematic: 3 TCP pos, xyz
        target: 3 Target_pos xyz
        # """

        number_of_agent_observations = self.robot_model.nq + 2*self.robot_model.nv
        self.observation_space = gym.spaces.Dict(
            {
                #Joint angles and join acc    
                "agent": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_agent_observations, ), dtype=np.float32), 

                #sensor data, tcp acc, (later on maybe IMU)
                "sensors": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.nsensor,), dtype = np.float32),

                #has to be determined by inverse kinematics in reality
                "tcp_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype = np.float32),

                #Target Pos
                "target": gym.spaces.Box(low=-np.inf,high=np.inf, shape=(3,), dtype=int),  # [x, y, z] coordinates
            }
        )

        # Define what actions are available 
        number_of_actuaters = self.robot_model.nu
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(number_of_actuaters,), dtype=np.float32),   # motor drehmoment


        
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        agent_obs = np.concatenate([self.data.qpos.flat[:], 
                                    self.data.qvel.flat[:], 
                                    self.data.qacc.flat[:]]).astype(np.float32)
        
        sensor_obs = np.concatenate([self.data.sensor("gyro").data.flat[:], 
                                    self.data.sensor("accel").data.flat[:]]).astype(np.float32)

        tcp_pos_obs = self.data.site("tcp").xpos.copy()

        return {"agent": agent_obs,
                "sensor": sensor_obs,
                "tcp_pos": tcp_pos_obs,
                "target": self.target_pos}
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info about KPIs like distance, energy...
        """
        return {
            self.info
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # start the robot in a random configuration(radnom pos and speed)
        rng = np.random.default_rng()
        #get joint_limits
        joint_range_limits = self.robot_model.jnt_range #[[low1,high1][low2,high2]]
        random_pos = rng.uniform(joint_range_limits[:,0], joint_range_limits[:,1], size=self.robot_model.nv)
        
        random_vel = rng.integers(low = 0.1, high= 1, size= self.robot_model.nv)
        #write new positions to data with [:]
        self.data.qpos[:] = random_pos
        self.data.qvel[:] = random_vel
        mujoco.mj_forward(self.model, self.data)

        # Randomly place target, ensuring it's different from tcp pos
        geoms = self.data.geom_size
        lenghts = [np.max(geom) for geom in geoms]
        max_range = sum(lenghts)
        self.target_pos = rng.uniform(low=0.1, high= 0.8*max_range, size= 3)
        #if tcp pos and target are equal change target
        # FIXME: dont compare pos, change to distance greater than
        while np.array_equal(self.target_pos, self.data.site("tcp").xpos):
            self.target_pos = rng.uniform(low=0.1, high= 0.8*max_range, size= 3)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take 

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the action to motor torque

        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        
        distance = np.linalg.norm(self._agent_location - self._target_location)
        energy = ...
        time = ...

        self.info = {"distance": distance,
                     "energy": energy,
                     "time": time
                    }
        

        reward = self.calculate_reward(self.info)
        observation = self._get_obs()
        info = self.info

        return observation, reward, terminated, truncated, info
    
    def calculate_reward(self, measurements: dict):
            reward = measurements
            return reward