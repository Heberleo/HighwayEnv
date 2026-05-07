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

class AccEnv(AbstractEnv):
    """A highway environment with continuous action space."""
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "attributes": ["x", "y", "vx", "vy"],
                    "see_behind": True,
                    "relative": True,
                    "normalize": True,
                    "vehicles_count": 2,  # Number of vehicles in each lane to observe (including the ego vehicle)
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-0.25, 0.25],
                    "acceleration_range": [-5, 5],
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

                "max_speed": 30,  # [m/s]

                "duration": 40,  # [s]

                "initial_speed": 0,

                "other_speed": 10,  # [m/s]
                

                "target_distance": 10,  # [m]
                
                "ego_length": 5,
                "other_length": 5,
                
                "distance_reward": 0.5,  # Penalty for being far from the target distance to the front vehicle
                "distance_norm": 20,  # Normalization factor for distance penalty
                "off_road_penalty": -1.0,  # Penalty for being off the road 
                "collision_penalty": -1.0,  # Penalty for colliding with another vehicle


            }
        )
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        
    
    def _create_road(self):
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(1, speed_limit=30),
            np_random=self.np_random
        )

    def _create_vehicles(self):
        """Create vehicles as specified in the configuration."""

        vehicle = Vehicle.create_random(
            self.road,
            lane_id=None,
            speed=self.config["initial_speed"],
        )

        vehicle = self.action_type.vehicle_class(
            self.road, vehicle.position, vehicle.heading, vehicle.speed
        )
        vehicle.LENGTH = self.config["ego_length"]

        # Cap ego vehicles speed 
        vehicle.MAX_SPEED = self.config["max_speed"]  
        vehicle.MIN_SPEED = 0

        self.road.vehicles.append(vehicle)
        self.vehicle = vehicle

        # Create a front vehicle at the target distance
        front_vehicle = Vehicle.create_at(
            road=self.road,
            lane_id=vehicle.lane_index[2],
            x=vehicle.position + np.array([self.config["target_distance"], 0]),
            speed=self.config["other_speed"],
        )
        front_vehicle.LENGTH = self.config["other_length"]
        self.road.vehicles.append(front_vehicle)

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

        vehicle_ahead, _ = self.road.neighbour_vehicles(self.vehicle, lane_index=self.vehicle.lane_index)

        distance_to_front = vehicle_ahead.position[0] - self.vehicle.position[0] - self.vehicle.LENGTH / 2 - vehicle_ahead.LENGTH / 2 if vehicle_ahead else 20
        if distance_to_front < self.config["target_distance"]:
            distance_reward = 1 - (self.config["target_distance"] - distance_to_front) / self.config["target_distance"]
            distance_reward **= 2  # Quadratic penalty for distance deviation
        else:
            distance_reward = 1 - (distance_to_front - self.config["target_distance"]) / self.config["distance_norm"]

        return {
            "collision_penalty": float(self.vehicle.crashed),
            "off_road_penalty": float(not self.vehicle.on_road),
            "distance_reward": distance_reward
        }

   