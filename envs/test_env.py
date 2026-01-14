import gymnasium as gym
import numpy as np
import mujoco


class RobotWorldEnv(gym.MujocoEnv):
    
    def __init__(self, model_path: str):
    
        #load model from Path
        self.robot_model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.robot_model)
        

        # Define what the agent can observe
        # TCP Pos, TCP Acceleration, Joint Angles, Joint Acceleration, Target Pos
        # Dict space gives us structured, human-readable observations
        #nv = degrees of freedom
        #nq = number of generalized coords, das kracht beim continums robot, weil dann quateriums oder so benutzt werden
        number_of_observations = self.robot_model.nv + self.robot_model.nq + ...
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_joints,), dtype=float32),   # motor drehmoment

                "target": gym.spaces.Box(low=-np.inf,high=np.inf, shape=(3,), dtype=int),  # [x, y, z] coordinates
            }
        )

        # Define what actions are available 
        number_of_actuaters = self.robot_model.nu
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(number_of_actuaters,), dtype=np.float32),   # motor drehmoment


        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }


    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
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

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
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

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        distance = np.linalg.norm(self._agent_location - self._target_location)
        reward = 1 if terminated else -0.1 * distance

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info