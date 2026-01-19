import gymnasium as gym
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from typing import Optional
from mujoco import viewer



class RobotWorldEnv(gym.Env):
    
    def __init__(self, model_path: str, render_mode: Optional[str] = None, config: Optional[dict] = None):
    
        #load model from Path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.target_pos = np.array([1,1,1]) # just for initialition
        self.steps_passed = 0
        self.goal_distance = 0.1
        self.info = {}  #just for initaliation, later on gets filled with for tracking succes
        self.rewards = {} #might be useful for tracking later on
        self.max_energy = np.sum(np.square(self.model.actuator_ctrlrange[:,1]))

        self.viewer = None
        self.render_mode = render_mode
        self.max_steps = 10_000

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

        number_of_agent_observations = self.model.nq + 2*self.model.nv
        self.observation_space = gym.spaces.Dict(
            {
                #Joint angles, joint vel, join acc   
                "agent": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_agent_observations, ), dtype=np.float32), 

                #sensor data, tcp acc, (later on maybe IMU)
                "sensors": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.sensordata.shape[0],), dtype = np.float32),

                #has to be determined by inverse kinematics in reality
                "tcp_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype = np.float32),

                #Target Pos
                "target": gym.spaces.Box(low=-np.inf,high=np.inf, shape=(3,), dtype=np.float64),  # [x, y, z] coordinates
            }
        )

        # Define what actions are available 
        number_of_actuaters = self.model.nu
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(number_of_actuaters,), dtype=np.float32)   # motor drehmoment


        
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        agent_obs = np.concatenate([self.data.qpos.flat[:], 
                                    self.data.qvel.flat[:], 
                                    self.data.qacc.flat[:]]).astype(np.float32)
        
        sensors_obs = np.concatenate([self.data.sensor("gyro").data.flat[:], 
                                    self.data.sensor("accel").data.flat[:]]).astype(np.float32)

        tcp_pos_obs = self.data.site("tcp").xpos.copy().astype(np.float32)

        return {"agent": agent_obs,
                "sensors": sensors_obs,
                "tcp_pos": tcp_pos_obs,
                "target": self.target_pos}
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info about KPIs like distance, energy...
        """
        return self.info
    
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

        self.steps_passed = 0

        # start the robot in a random configuration(radnom pos and speed)
        #get joint_limits
        joint_range_limits = self.model.jnt_range #[[low1,high1][low2,high2]]
        random_pos = self.np_random.uniform(joint_range_limits[:,0], joint_range_limits[:,1], size=self.model.nv)
        
        random_vel = self.np_random.uniform(low = -1, high= 1, size= self.model.nv)
        #write new positions to data with [:]
        self.data.qpos[:] = random_pos
        self.data.qvel[:] = random_vel
        mujoco.mj_forward(self.model, self.data)

        # Randomly place target, ensuring it's different from tcp pos
        self.target_pos = self.calculate_target_for_sphere()
        #if tcp pos and target are equal change target
        tcp_pos = self.data.site("tcp").xpos
        distance = np.linalg.norm(tcp_pos - self.target_pos)
        while (distance <= self.goal_distance):
            self.target_pos = self.calculate_target_for_sphere()
            distance = np.linalg.norm(tcp_pos - self.target_pos) 

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
        self.steps_passed = self.steps_passed+1

        # Map the action to motor torque
        torque = action*self.model.actuator_ctrlrange[:,1]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        tcp_pos = self.data.site("tcp").xpos
        distance = np.linalg.norm(tcp_pos - self.target_pos)

        # Check if agent reached the target
        terminated = bool(distance < self.goal_distance)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = self.steps_passed > self.max_steps

        #doesnt accuratly calculate energy, but strongly correlates
        
        energy = np.sum(np.square(action))

        self.info = {
                    "tcp" : tcp_pos,
                    "target": self.target_pos,
                    "distance": distance,
                    "energy": energy,
                    "steps_passed": self.steps_passed
                    }
        
        #render if render_mode is specified, skip if none
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            self.render()

        reward = self.calculate_reward(self.info)
        observation = self._get_obs()
        info = self.info

        return observation, reward, terminated, truncated, info
    
    def calculate_reward(self, measurements: dict):
            self.rewards = {}
            
            #distance worth the most 
            self.rewards["distance_reward"] = -3*measurements["distance"]
            
            #energy  
            self.rewards["energy_reward"] = -0.1*(measurements["energy"]/self.max_energy)

            if measurements["distance"] < self.goal_distance:
                self.rewards["goal_reward"] = 50
            else:
                self.rewards["goal_reward"] = 0

            reward = sum(self.rewards.values())
            return reward
    
    def calculate_target_for_sphere(self):
        geoms = self.model.geom_size
        lenghts = [np.max(geom) for geom in geoms]
        max_range = sum(lenghts)
        phi = self.np_random.uniform(low=0, high= 2*np.pi, size=1)
        theta = self.np_random.uniform(low=0, high= np.pi/2, size=1)
        # HACK: theta set to 90° for planar, max radius is set to fixed value
        theta = np.pi/2 
        radius = self.np_random.uniform(low=0.1, high= 0.55, size=1)
        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(theta)
        return np.array([x,y,z]).flatten()
    
    def get_rewards(self):
        return self.rewards
    
    def render(self):
        """Full chatty, keine ahnung was hier passiert"""
        # Minimaler Render-Code für gym.Env
        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

        if self.render_mode == "human":
            if self.viewer is None:
                # Startet den passiven Viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            
            self.viewer.sync()
            
            # visalize target with movable body named target
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

            if body_id != -1:
                # Die Mocap-ID holen (Mapping von Body -> Mocap)
                mocap_id = self.model.body_mocapid[body_id]

            if mocap_id != -1:
                self.data.mocap_pos[mocap_id] = self.target_pos 
                # update kinematics
                mujoco.mj_forward(self.model, self.data) 