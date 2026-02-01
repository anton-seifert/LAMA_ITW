import gymnasium as gym
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from typing import Optional
from mujoco import viewer



class RobotWorldEnv(gym.Env):
    
    def __init__(self, config: Optional[dict] = None, render_mode: Optional[str] = None, ):
        print("creating 3DOF_env...")
        #LOAD Variables from config
        #first value = key from dict, second value fallback default value
        self.model_path = config.get("robot_model_path")
        self.goal_distance = config.get("goal_distance", 0.1)
        self.max_steps = config.get("max_steps", 1_000)
        self.distance_reward_factor = config.get("distance_reward", 20)
        self.energy_reward_factor = config.get("energy_reward", 0.2)
        self.goal_reward_factor = config.get("goal_reward", 50)
        self.truncated_distance_steps = config.get("truncated_distance_steps")
        self.truncated_distance_reward_factor = config.get("truncated_distance_reward")
        self.duration_in_target = config.get("duration_in_target")
        self.steps_in_range_reward = config.get("in_range_reward")
        

        #load model from Path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.target_pos = np.array([1,1,1]) # just for initialition
        self.steps_passed = 0
        self.steps_passed_in_goal_range_total = 0
        self.info = {}  #just for initaliation, later on gets filled with for tracking succes
        self.rewards = {} #might be useful for tracking later on
        self.max_energy = np.sum(np.square(self.model.actuator_ctrlrange[:,1]))

        self.viewer = None
        self.render_mode = render_mode

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
        #print(f"number of agent observations {number_of_agent_observations}")
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
        number_of_actuators = self.model.nu
        #print(f"number of actuators {number_of_actuators}")
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(number_of_actuators,), dtype=np.float32)   # motor drehmoment normalized


        
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
        self.steps_passed_in_goal_range_total = 0

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
        tcp_pos = self.data.site("tcp").xpos
        distance = np.linalg.norm(tcp_pos - self.target_pos)
        #if tcp pos and target are too close, look for new target
        while (distance <= 0.5):
            #print("calc new target")
            self.target_pos = self.calculate_target_for_sphere()
            distance = np.linalg.norm(tcp_pos - self.target_pos)

        self.info["best_distance"] = distance
        self.info["best_distance_step"] = 0
        self.info["in_target_range_step"] = 0
        self.info["reached_target"] = False

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
        terminated = False
        # Map the action to motor torque
        torque = action*self.model.actuator_ctrlrange[:,1]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        tcp_pos = self.data.site("tcp").xpos
        distance = np.linalg.norm(tcp_pos - self.target_pos)

        reached_target = False
        
        # Check if agent reached the target
        if (distance < self.goal_distance):
            reached_target = True
            self.steps_passed_in_goal_range_total = self.steps_passed_in_goal_range_total+1
            #only start counting steps in target range if before wansnt in range
            if(self.info["reached_target"] == False):
                self.info["in_target_range_step"] = self.steps_passed
        else:
            reached_target = False
        
        #terminates if tcp stayed in target range for set duration of steps
        if((self.steps_passed - self.info["in_target_range_step"] >= self.duration_in_target) and (reached_target == True)):
            terminated = True


        # Truncated after set step limit
        truncated = self.steps_passed > self.max_steps

        #Updates best(smallest) distance
        if(distance < self.info["best_distance"]):
            self.info["best_distance"] = distance
            self.info["best_distance_step"] = self.steps_passed

        #terminates if distance didnt get smaller after set time steps
        truncated_distance = False
        if((self.steps_passed - self.info["best_distance_step"] >= self.truncated_distance_steps) and (reached_target == False)):
            truncated_distance = True
            truncated = True


        #doesnt accuratly calculate energy, but strongly correlates
        energy = np.sum(np.square(action))

        self.info.update({
                    "tcp" : tcp_pos,
                    "target": self.target_pos,
                    "distance": distance,
                    "energy": energy,
                    "steps_passed": self.steps_passed,
                    "reached_target": reached_target,
                    "truncated_distance": truncated_distance,
                    "terminated" : terminated,
                    "total_steps_passed_in_goal_range" :  self.steps_passed_in_goal_range_total
                    })
        
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
            self.rewards["distance_reward"] = -self.distance_reward_factor*measurements["distance"]
            
            #energy  
            self.rewards["energy_reward"] = -self.energy_reward_factor*(measurements["energy"]/self.max_energy)

            #punish if gets truncated beacuse of no distance improvemnts after set steps
            if(self.info["truncated_distance"]):
                self.rewards["truncated_distance"] = -self.truncated_distance_reward_factor
            else:
                self.rewards["truncated_distance"] = 0

            #reward for staying in goal space
            if(measurements["distance"] < self.goal_distance):
                self.rewards["in_range_reward"] = self.steps_in_range_reward
            else:
                self.rewards["in_range_reward"] = 0

            #reward for termination    
            if measurements["terminated"] == True:
                self.rewards["goal_reward"] = self.goal_reward_factor
            else:
                self.rewards["goal_reward"] = 0

            reward = sum(self.rewards.values())
            return reward
    
    def calculate_target_for_sphere(self):
        geoms = self.model.geom_size
        lenghts = [np.max(geom) for geom in geoms]
        max_range = sum(lenghts)
        phi = self.np_random.uniform(low=0, high= 2*np.pi)
        theta= 0
        if(self.model_path == "assets/test_robot.xml"):
            theta = np.pi/2 
        elif(self.model_path == "assets/test_robot_3DOF.xml"):
            theta = self.np_random.uniform(low=0, high= np.pi/2)
        #HACK: theta set to 90° for planar z = 0, max radius is set to fixed value
        
        radius = self.np_random.uniform(low=0.1, high= 0.55)
        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(theta)
        return np.array([x,y,z]).flatten()
    
    def get_rewards(self):
        return self.rewards
    
    def render(self):
        """Full chatty, keine ahnung was hier passiert"""
        #------
        # diesen code block nach unten verschieben, wenn man das target immer in der gleichen stelle haben will

        # visalize target with movable body named target
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if body_id != -1:
            # Die Mocap-ID holen (Mapping von Body -> Mocap)
            mocap_id = self.model.body_mocapid[body_id]

        if mocap_id != -1:
            self.data.mocap_pos[mocap_id] = self.target_pos 
            # update kinematics
            mujoco.mj_forward(self.model, self.data) 
        #---------------

        # Minimaler Render-Code für gym.Env
        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

        if self.render_mode == "human":
            if self.viewer is None:
                # Startet den passiven Viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            
            self.viewer.sync()
            
        

        

    def _get_rgb_array(self):
        """ Full Chatty, keine Ahnung was hier geht"""
    # Renderer wird nur beim ersten Aufruf erstellt (Lazy Loading)
        if not hasattr(self, 'renderer'):
        # WICHTIG: width/height bestimmen die Video-Auflösung
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    # Die aktuelle Physik-Szene in den Renderer laden
        self.renderer.update_scene(self.data) 
    # Falls du eine bestimmte Kamera hast: update_scene(self.data, camera="kamera_name")
    
    # Das Bild als Numpy-Array zurückgeben
        return self.renderer.render()
    

    """fully chatty"""
    def close(self):
    # Renderer löschen, falls er existiert
        if hasattr(self, 'renderer'):
            del self.renderer
    
    # Viewer schließen (hast du wahrscheinlich schon)
        if self.viewer is not None:
            self.viewer.close()
        super().close()