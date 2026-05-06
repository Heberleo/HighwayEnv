from __future__ import annotations
from turtle import speed

import numpy as np

from highway_env import utils
from highway_env import road
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road import lane
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import lmap, near_split
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

class ContinuousSimpleEnv(AbstractEnv):
    """A highway environment with continuous action space."""
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "attributes": ["x", "y", "vx", "vy", "lat_off, long_off, ang_off", "rightness"],
                    "see_behind": True,
                    "relative": True,
                    "normalize": True,
                    "vehicles_count": 1,  # Number of vehicles in each lane to observe (including the ego vehicle)
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-0.25, 0.25],
                    "acceleration_range": [-2, 2],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": False,
                },
                "simulation_frequency": 10,
                "policy_frequency": 1,
                "screen_width": 1200,
                "screen_height": 300,
                "scaling": 7,
                "centering_position": [0.3, 0.5],

                "max_speed": 15,  # [m/s]

                "duration": 40,  # [s]

                "initial_speed_range": [0, 15],
                
                "lanes_count": 5,

                "target_speed": 10,  # [m/s]
                "target_lane": 0,  # Index of the lane the agent should aim to be in (0 is leftmost)
                
                "speed_reward": 0.5,  # Reward for driving at the target speed
                "speed_penalty": -1.0,  # Penalty for driving too slow or too fast
                "heading_penalty": -1.0,  # Penalty for heading deviation from lane direction
                "lateral_penalty": -0.5,  # Penalty for being far from the lane center
                "off_road_penalty": -5.0,  # Penalty for being off the road 
                "acceleration_penalty": -0.0,  # Penalty for high acceleration to foster smooth driving
                "target_lane_penalty": -0.0,  # Penalty for not being in the target lane
                "target_lane_reward": 0.5,  # Reward for being in the target lane
            }
        )
        return config
    
    def _reset(self) -> None:
        self.vehicles_behind_last_step = 0

        self._create_road()
        self._create_vehicles()
        
    
    def _create_road(self):
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=15),
            np_random=self.np_random
        )

    def _create_vehicles(self):
        """Create vehicles as specified in the configuration."""

        speed = self.np_random.uniform(*self.config["initial_speed_range"])
        vehicle = Vehicle.create_random(
            self.road,
            lane_id=None,
            speed=speed,
        )

        if vehicle.lane_index[2] == 0: 
            heading = self.np_random.uniform(0, np.pi/4)  # Random initial heading within a reasonable range
        elif vehicle.lane_index[2] == self.config["lanes_count"] - 1:
            heading = self.np_random.uniform(-np.pi/4, 0)  # Random initial heading within a reasonable range
        else:
            heading = self.np_random.uniform(-np.pi/4, np.pi/4)  # Random initial heading within a reasonable range

        vehicle = self.action_type.vehicle_class(
            self.road, vehicle.position, heading, vehicle.speed
        )

        # Cap ego vehicles speed 
        vehicle.MAX_SPEED = self.config["max_speed"]  
        vehicle.MIN_SPEED = 0

        self.road.vehicles.append(vehicle)
        self.vehicle = vehicle

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform an action, advance the simulation, and keep surrounding traffic populated.

        This mirrors the base environment step logic, but refreshes nearby traffic after the
        simulation step so vehicles that drift too far away are pruned and new vehicles are
        spawned ahead of / behind the controlled vehicle(s).
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]

        action = self._disturb_action(action)
        self._simulate(action)
        
        obs = self.observation_type.observe()
        obs = self._disturb_observation(obs)

        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        info["true_observation"] = obs
        info["on_road"] = self.vehicle.on_road

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def _disturb_action(self, action: Action) -> Action:

        """
        Action Disturbances:

        - lateral/ longitudinal wind: A constant bias can be added to the steering/acceleration action to 
        simulate the effect of wind pushing.

        - steering/ acceleration factor: The actual steering/ acceleration applied can be a scaled version 
        of the action taken by the agent, simulating a situation where the vehicle is more or less 
        responsive to control inputs.

        """
        steering_offset = self.config.get("steering_offset", 0.0)
        acceleration_offset = self.config.get("acceleration_offset", 0.0)
        steering_factor = self.config.get("steering_factor", 1.0)
        acceleration_factor = self.config.get("acceleration_factor", 1.0)

        # map offsets to [-1, 1] range, so that they can be added to the action which is in this range
        steering_offset = lmap(steering_offset, self.action_type.steering_range, [-1, 1])
        acceleration_offset = lmap(acceleration_offset, self.action_type.acceleration_range, [-1, 1]) - lmap(0., self.action_type.acceleration_range, [-1, 1]) # accomodate for asymmetric acceleration range

        disturbed_steering = np.clip(action[1] * steering_factor + steering_offset, -1, 1)
        disturbed_acceleration = np.clip(action[0] * acceleration_factor + acceleration_offset, -1, 1)

        action = np.array([disturbed_acceleration, disturbed_steering])

        return action
    
    def _disturb_observation(self, observation: np.ndarray) -> np.ndarray:

        """
        Observation Disturbances:

        - scale: The observed values can be scaled to simulate sensor calibration issues.

        - offset: A constant offset can be added to the observed values to simulate sensor bias.

        - distortion: The observed values can be distorted using a non-linear function to simulate sensor distortion.

        """


        return observation

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        return reward

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _rewards(self, action) -> dict[str, float]:
        """Compute all individual reward components."""
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)

        target_speed = self.config["target_speed"]

        speed_diff = abs(forward_speed - target_speed)
        SPEED_NORM = self.vehicle.MAX_SPEED - self.vehicle.MIN_SPEED
        if speed_diff < 0.1:
            speed_reward = 1.0
            speed_penalty = 0.0
        else:
            speed_reward = 0.0
            speed_penalty = np.clip(speed_diff / SPEED_NORM, 0, 1)

        heading_penalty, lateral_penalty, target_lane_penalty = self._lane_penalties()

        acceleration = action[0] # Acceleration action is in range [-1, 1], where -1 corresponds to max deceleration and 1 to max acceleration
        acceleration_punished = np.clip(acceleration, 0, 1) # Only penalize positive acceleration, square it to have a stronger penalty for higher accelerations


        return {
            "speed_reward": speed_reward,
            "speed_penalty": speed_penalty,
            "off_road_penalty": float(not self.vehicle.on_road),
            "heading_penalty": heading_penalty,
            "lateral_penalty": lateral_penalty,  # Penalize being in the leftmost or rightmost lane to foster lane keeping
            "acceleration_penalty": acceleration_punished,  # Penalize high acceleration to foster smooth driving
            "target_lane_penalty": target_lane_penalty,  # Penalty for not being in the target lane
            "target_lane_reward": 1 - target_lane_penalty,  # Reward for being in the target lane
        }

    def _lane_penalties(self) -> tuple[float, float, float]:
        # 1. Get raw angles (in radians)
        vehicle_heading = self.vehicle.heading
        
        # 2. Get the lane's heading at the vehicle's current position (actually, in this road network setup, the lane heading is constant)
        lane = self.vehicle.lane
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        lane_heading = lane.heading_at(longitudinal)
        
        # 3. Calculate the shortest angular difference (handles the 360-degree wrap-around)
        # This guarantees the difference is strictly between -pi and +pi
        angle_diff = (vehicle_heading - lane_heading + np.pi) % (2 * np.pi) - np.pi
        
        # 4. Calculate the Cosine Penalty
        # cos(0) = 1 (perfect alignment). cos(90 deg) = 0.
        # Therefore, 1 - cos(theta) gives us 0 for perfect, and 1 for a 90-degree swerve.
        heading_penalty = 1.0 - np.cos(angle_diff)

        # 5. Calculate penalty for being far from the lane center (lateral deviation)
        # This encourages the vehicle to stay centered in the lane, which is important for safety and realism.
        lateral_distance = lane.distance(self.vehicle.position)

        width = lane.width_at(longitudinal)
        lateral_penalty = (lateral_distance / (width / 2)) ** 2


        target_lane = self.config["target_lane"]
        lane_index = self.vehicle.lane_index
        neighbours = self.road.network.all_side_lanes(lane_index)
        target_lane_penalty = abs(lane_index[2] - target_lane) / max(len(neighbours) - 1, 1)



        return heading_penalty, lateral_penalty, target_lane_penalty

    