import gymnasium as gym
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from typing import Optional
from mujoco import viewer
import time



class RobotWorldEnv(gym.Env):
    
    def __init__(self, config: Optional[dict] = None, render_mode: Optional[str] = None, target_angle: Optional[float] = None):
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
        self.joint_distance_reward_factor = config.get("joint_limit_reward")
        self.duration_in_target = config.get("duration_in_target")
        self.steps_in_range_reward = config.get("in_range_reward")
        self.singularity_reward_factor = config.get("singularity_reward_factor")
        self.crash_reward_factor = config.get("crash_reward")
        self.floor_distance_reward_factor = config.get("floor_distance_reward")
        self.target_angle = target_angle

        self.options = None

        #load model from Path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.floor_id = self.model.geom("floor").id
        self.link1_id = self.model.site("site_link1").id
        self.tcp_id = self.model.site("tcp").id
        self.gyro_id = self.model.sensor("gyro").id
        self.accel_id = self.model.sensor("accel").id
        self.floor_level = self.model.geom_pos[self.floor_id][2]

        self.target_pos = np.array([])
        self.steps_passed = 0
        self.steps_passed_in_goal_range_total = 0
        self.info = {}  #just for initaliation, later on gets filled with for tracking succes
        self.rewards = {} #might be useful for tracking later on
        #self.max_energy = np.sum(np.square(self.model.actuator_ctrlrange[:,1]))
        self.last_action = np.array([0,0,0])
        self.total_distance = 0

        self.viewer = None
        self.render_mode = render_mode

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations

        # TCP Pos, TCP Acceleration, Joint Angles, Joint Velocity, Joint Acceleration, Target Pos

        """
        nv = degrees of freedom, also eig die number of velocities, or acceleration
        nq = number of generalized coords, die angle position (das kracht beim continums robot, weil dann quateriums oder so benutzt werden, nicht 3 komponenten, sondern 4)
        agent: 3 pos, 3 vel, 3 acc, 3 action
        sensor 3 TCP Acc, 3 TCP Gyro
        kinematic: 3 TCP pos, xyz
        target: 3 Target_pos xyz
        # """

        number_of_agent_observations = self.model.nq + 3*self.model.nv
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

                #distance between TCP and Target
                "distance": gym.spaces.Box(low=-np.inf,high=np.inf, shape=(3,), dtype=np.float64),  # [x, y, z] coordinates

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
                                    self.data.qacc.flat[:],
                                    self.last_action.flat[:]]).astype(np.float32)
        
        sensors_obs = np.concatenate([self.data.sensordata]).astype(np.float32)

        tcp_pos_obs = self.data.site_xpos[self.tcp_id].copy().astype(np.float32)

        distance_obs = self.target_pos-tcp_pos_obs           
                                    

        return {"agent": agent_obs,
                "sensors": sensors_obs,
                "tcp_pos": tcp_pos_obs,
                "target": self.target_pos,
                "distance" : distance_obs}
    
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
        mujoco.mj_resetData(self.model, self.data)  # hard reset full simulator state


        self.steps_passed = 0
        self.steps_passed_in_goal_range_total = 0
        self.last_action = np.array([0,0,0])

        # start the robot in a random configuration(radnom pos and speed)
        #get joint_limits
        if(self.options is None):
            for i in range(100):
                margin = np.deg2rad(46)
                joint_range_limits = self.model.jnt_range #[[low1,high1][low2,high2]]
                #generate reandom staring pos with margin, high-maring, low +margin
                random_pos = self.np_random.uniform(low = joint_range_limits[:,0]+margin, high= joint_range_limits[:,1]-margin, size=self.model.nv)
                
                random_vel = self.np_random.uniform(low = -1, high= 1, size= self.model.nv)
                #write new positions to data with [:]
                self.data.qpos[:] = random_pos
                self.data.qvel[:] = random_vel
                mujoco.mj_forward(self.model, self.data)

                #if configuration is okay, break out of for loop
                if(self.check_floor_collision() == False):
                    break

            else: #run if foor loop doesnt break
                print("no random position found")
                pos = np.array([0, -1.7 , 2])
                random_vel = self.np_random.uniform(low = -1, high= 1, size= self.model.nv)

                self.data.qpos[:] = pos
                self.data.qvel[:] = random_vel
                mujoco.mj_forward(self.model, self.data)
        else:
            #print("starting from options")
            self.data.qpos[:] = self.options["start_pos"]
            self.data.qvel[:] = self.options["start_vel"]
            mujoco.mj_forward(self.model, self.data)




        # Randomly place target, ensuring it's different from tcp pos
        tcp_pos = self.data.site("tcp").xpos
        if(self.options is None):
            self.target_pos = self.calculate_target_for_sphere()
            distance = np.linalg.norm(tcp_pos - self.target_pos)
            #if tcp pos and target are too close, look for new target
            while (distance <= 0.5):
                #print("calc new target")
                self.target_pos = self.calculate_target_for_sphere()
                distance = np.linalg.norm(tcp_pos - self.target_pos)
        else:
            #print("target from options")
            self.target_pos = self.options["target_pos"]

        #rendering target sphere
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if body_id != -1:
            # Die Mocap-ID holen (Mapping von Body -> Mocap)
            mocap_id = self.model.body_mocapid[body_id]

        if mocap_id != -1:
            self.data.mocap_pos[mocap_id] = self.target_pos 
            # update kinematics
             

        mujoco.mj_forward(self.model, self.data)
        

        
        self.info["best_distance_step"] = 0
        self.entry_in_goal_space_step = 0
        self.info["reached_target"] = False
        self.total_distance = 0
        self.info["start_pos"] = self.data.qpos.tolist()
        self.info["start_vel"] = self.data.qvel.tolist()
        self.info["target"] = self.target_pos

        #warm up
        for _ in range(5):
            #calculates generalized forces to keep robot stable while warm up
            self.data.ctrl[:] = self.data.qfrc_bias[:self.model.nu]
            mujoco.mj_step(self.model, self.data)

        tcp_pos = self.data.site("tcp").xpos
        distance = np.linalg.norm(tcp_pos - self.target_pos)
        self.info["best_distance"] = distance

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

        tcp_pos = self.data.site_xpos[self.tcp_id]
        distance = np.linalg.norm(tcp_pos - self.target_pos)
        reached_target = False

        self.total_distance += distance

        #check for collision
        if(self.check_floor_collision() == True):
            self.info["floor_crash"] = True
            terminated = True
        else:
            self.info["floor_crash"] = False
        

        
        # Check if agent reached the target
        if (distance < self.goal_distance):
            reached_target = True
            self.steps_passed_in_goal_range_total = self.steps_passed_in_goal_range_total+1
            #only start counting steps in target range if before wansnt in range
            if(self.info["reached_target"] == False):
                self.entry_in_goal_space_step = self.steps_passed
        else:
            reached_target = False
        
        #terminates if tcp stayed in target range for set duration of steps
        if((self.steps_passed - self.entry_in_goal_space_step >= self.duration_in_target) and (reached_target == True)):
            stayed_in_target = True
            terminated = True
        else:
            stayed_in_target = False

        # Truncated after set step limit
        truncated = self.steps_passed >= self.max_steps

        #Updates best(smallest) distance
        if(distance < self.info["best_distance"]):
            self.info["best_distance"] = distance
            self.info["best_distance_step"] = self.steps_passed

        #terminates if distance didnt get smaller after set time steps
        truncated_distance = False
        if((self.steps_passed - self.info["best_distance_step"] >= self.truncated_distance_steps) and (reached_target == False)):
            truncated_distance = True
            truncated = True


        #calculate energy difference to last step, should lead to smooth actions
        if self.last_action is not None:
            energy = np.sum(np.square(self.last_action - action))
            self.last_action = action

        else:
            energy = np.sum(np.square(action))
            self.last_action = action


        self.info.update({
                    "tcp" : tcp_pos,
                    "distance": distance,
                    "total_distance" : self.total_distance,
                    "energy": energy,
                    "steps_passed": self.steps_passed,
                    "reached_target": reached_target,
                    "truncated_distance": truncated_distance,
                    "stayed_in_target" : stayed_in_target,
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
            
            #distance scales with steps passed
            # HACK: should scale with timesteps, better with timesteps passed without entering goal space
            
            self.rewards["distance_reward"] = -self.distance_reward_factor*np.log(measurements["distance"])

            #crash, contact with floor
            if(self.info["floor_crash"] == True):
                self.rewards["floor_crash_reward"] = -self.crash_reward_factor
            else:
                self.rewards["floor_crash_reward"] = 0

            #punishment for getting close to floor
            exponential_floor_distance = -self.floor_distance_reward_factor*np.exp(-self.calculate_floor_distance())
            self.rewards["floor_distance"] = np.sum(exponential_floor_distance)
            
            #energy  
            self.rewards["energy_reward"] = -self.energy_reward_factor*measurements["energy"]*10*np.exp(-measurements["distance"])-(measurements["energy"]*self.energy_reward_factor)

            #singularity punishment
            self.rewards["singularity_reward"] = -np.sum(np.square(self.data.qvel)) * self.singularity_reward_factor

            #punish if gets truncated beacuse of no distance improvemnts after set steps
            if(self.info["truncated_distance"]):
                self.rewards["truncated_distance"] = -self.truncated_distance_reward_factor
            else:
                self.rewards["truncated_distance"] = 0

            #punishment for getting close to joint limits
            exponential_joint_distance = -self.joint_distance_reward_factor*np.exp(-self.calculate_joint_limit_distance())
            self.rewards["joint_limits"] = np.sum(exponential_joint_distance)

            #reward for staying in goal space
            if(measurements["distance"] < self.goal_distance):
                self.rewards["in_range_reward"] = self.steps_in_range_reward
            else:
                self.rewards["in_range_reward"] = 0

            #reward for termination    
            if measurements["stayed_in_target"] == True:
                self.rewards["goal_reward"] = self.goal_reward_factor
            else:
                self.rewards["goal_reward"] = 0

            reward = sum(self.rewards.values())
            return reward
    
    def calculate_joint_limit_distance(self):
        joint_limits = (self.model.jnt_range) #[[low, high]]
        joint_pos = (self.data.qpos)#[pos1,pos2,...]

        joint_distance = np.min(np.abs(joint_limits-joint_pos[:,None]), axis=1)
        return joint_distance
    
    def check_floor_collision(self):
        contacts = self.data.contact

        for contact in contacts:
            if contact.geom1 == self.floor_id or contact.geom2 == self.floor_id:
                return True
        else:
            return False
        
    def calculate_floor_distance(self):
        
        #get z coords of ends of links
        link1_pos = self.data.site_xpos[self.link1_id][2]
        link2_pos = self.data.site_xpos[self.tcp_id][2]
        #calc dist to floor for each link
        #floor_dist_link1 = (np.abs(floor_level - link1_pos))
        #floor_dist_link2 = (np.abs(floor_level - link2_pos))
        #creates array of distances beteween links and floor
        distances = np.array([np.abs(self.floor_level - link1_pos), np.abs(self.floor_level - link2_pos)])
        return distances

    
    def calculate_target_for_sphere(self):
        geoms = self.model.geom_size
        lenghts = [np.max(geom) for geom in geoms]
        max_range = sum(lenghts)
        phi = self.np_random.uniform(low=0, high= self.target_angle)
        theta= 0
        if(self.model_path == "assets/test_robot.xml"):
            theta = np.pi/2 
        elif(self.model_path == "assets/test_robot_3DOF.xml"):
            theta = self.np_random.uniform(low=0.07, high= np.pi/2)
        #HACK: theta set to 90° for planar z = 0, max radius is set to fixed value
        
        radius = self.np_random.uniform(low=0.15, high= 0.55)
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
        
        #---------------

        # Minimaler Render-Code für gym.Env
        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

        if self.render_mode == "human":
            if self.viewer is None:
                # Startet den passiven Viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def set_reset_options(self, options: dict):
        self.options = options
            
        

        

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