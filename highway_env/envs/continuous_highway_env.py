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

class ContinuousHighwayEnv(AbstractEnv):
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
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-0.35, 0.35],
                    "acceleration_range": [-5, 5],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": False,
                },
                "simulation_frequency": 10,
                "policy_frequency": 5,
                "screen_width": 1200,
                "screen_height": 300,
                "scaling": 7,
                "centering_position": [0.3, 0.5],

                "max_speed": 10,  # [m/s]

                "duration": 40,  # [s]

                "ego_initial_speed": 7.0,
                "reward_speed_range": [5, 10],
                "penalty_speed_range": [0, 5],
                "speeding_range": [10, 12],

                # road and traffic
                "traffic": None,
                "lanes_count": 4,
                "vehicles_count": 10,
                "others_speed_range": [5, 10],
                "initial_lane_id": None,  # If None, a random lane is sampled
                "initial_heading": None,  # If None, the lane heading is used, if "random", a random heading is sampled

                "right_lane_reward": 0.0,  # Reward for being in the rightmost lane
                "speeding_penalty": -0.2,  # Penalty for driving above the speed limit
                "high_speed_reward": 0.2,  # Reward for driving at high speed
                "trailing_penalty": -0.2,  # Penalty for being too close to the front vehicle
                "low_speed_penalty": -0.2,  # Penalty for driving at low speed
                "collision_reward": -1.0,  # Penalty for collisions
                "heading_penalty": -2.0,  # Penalty for heading deviation from lane direction
                "lateral_penalty": -0.5,  # Penalty for being far from the lane center
                "off_road_penalty": -1.0,  # Penalty for being off the road 
                "acceleration_penalty": -0.1,  # Penalty for high acceleration to foster smooth driving
            }
        )
        return config
    
    def _reset(self) -> None:
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
        vehicle = Vehicle.create_random(
            self.road,
            lane_id=self.config["initial_lane_id"],
            speed=self.config["ego_initial_speed"]
        )
        heading = vehicle.heading
        if self.config["initial_heading"] == "random":
            heading = self.np_random.uniform(-np.pi/4, np.pi/4)  # Random initial heading within a reasonable range

        vehicle = self.action_type.vehicle_class(
            self.road, vehicle.position, heading, vehicle.speed
        )

        # Cap ego vehicles speed 
        vehicle.MAX_SPEED = self.config["max_speed"]  
        vehicle.MIN_SPEED = 0

        self.road.vehicles.append(vehicle)
        self.vehicle = vehicle

        if self.config["traffic"] is None:
            return  # No traffic to add
        elif self.config["traffic"] == "slalom":
            self._slalom_traffic() 
        elif self.config["traffic"] == "dense_slalom":
            if self.np_random.uniform() < 0.75:  
                self._dense_slalom_traffic()  
            else:
                self._slalom_traffic() 

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
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.lane_index[2]

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        low_scaled_speed = utils.lmap(
            forward_speed, self.config["penalty_speed_range"], [0, 1]
        )
        low_scaled_speed = 1 - low_scaled_speed  # We want a penalty for low speeds, so we invert the scale
    
        heading_penalty, lateral_penalty, trailing_penalty = self._lane_penalties()

        speeding = utils.lmap(
            forward_speed, self.config["speeding_range"], [0, 1]
        )

        acceleration = action[0] # Acceleration action is in range [-1, 1], where -1 corresponds to max deceleration and 1 to max acceleration
        acceleration_punished = np.clip(acceleration, 0, 1) # Only penalize positive acceleration, square it to have a stronger penalty for higher accelerations

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "low_speed_penalty": np.clip(low_scaled_speed, 0, 1),
            "off_road_penalty": float(not self.vehicle.on_road),
            "heading_penalty": heading_penalty,
            "lateral_penalty": lateral_penalty,  # Penalize being in the leftmost or rightmost lane to foster lane keeping
            "speeding_penalty": np.clip(speeding, 0, 1),  # Penalize driving above the speed limit
            "acceleration_penalty": acceleration_punished,  # Penalize high acceleration to foster smooth driving
            "trailing_penalty": trailing_penalty  # Penalize being too close to the vehicle ahead
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

        # if abs(angle_diff) > 0.1: # apply no lateral penalty if the vehicle is turning
           # lateral_penalty = 0.0

        trailing_penalty = 0.0
        vehicle_ahead = self.road.neighbour_vehicles(self.vehicle, lane_index=self.vehicle.lane_index)[0]
        if vehicle_ahead is not None:
            distance = self.vehicle.lane_distance_to(vehicle_ahead)

            if distance < 10 and abs(angle_diff) < 0.1: # trailing penalty unless the vehicle is turning, in which case we assume it is performing a lane change and we don't want to penalize it for being close to the front vehicle
                trailing_penalty = (10 - distance) / 10  # Linear penalty that increases as the vehicle gets closer to the front vehicle, maxing out at a distance of 0
        
        
        return heading_penalty, lateral_penalty, trailing_penalty
    
    def _dense_slalom_traffic(self):
        num_lanes = self.config["lanes_count"]
        lanes = self.road.network.lanes_list()

        num_vehicles = 16
        ego_position = self.vehicle.position
        distance = 40
        speed = self.np_random.uniform(*self.config["others_speed_range"])  # Random speed for vehicles in the dense slalom traffic to create more diverse traffic patterns
        counter = 0

        max_rows = num_vehicles // 2
 
        pair_probability = 0.75  # Probability of spawning a pair of vehicles in adjacent lanes

        while counter < max_rows:
            # create random permutation of lane indices for this batch of vehicles
            lane_index = self.np_random.choice(num_lanes)

            # spawn a vehicle in the current lane
            lane = lanes[lane_index]
            x = lane.position(ego_position[0], ego_position[1])[0]  # spawn behind the ego vehicle
            x += distance + counter * distance  # space out vehicles by a certain distance
            x += self.np_random.uniform(-10, 10)  # Add some randomness to the position of vehicles in the slalom traffic to make it more dynamic and less predictable
            
            random_speed = speed + self.np_random.uniform(-0.1, 0.1)  # Add some randomness to the speed of vehicles in the slalom traffic to make it more dynamic and less predictable

            counter += 1
            vehicle = Vehicle(self.road, lane.position(x, 0), lane.heading_at(0), random_speed)  # Vehicles in the slalom traffic cannot change lane, to foster the idea of a "slalom"
            self.road.vehicles.append(vehicle)

            if self.np_random.uniform() < pair_probability:  # Spawn a pair of vehicles in adjacent lanes with a certain probability to create more challenging traffic patterns
                adjacent_lane_index = (lane_index + 1) % num_lanes  # Get the adjacent lane index (wrap around)
                adjacent_lane = lanes[adjacent_lane_index]
                x_adjacent = x + self.np_random.uniform(5, 10)  # Add some randomness to the position of vehicles in the slalom traffic to make it more dynamic and less predictable

                vehicle_adjacent = Vehicle(self.road, adjacent_lane.position(x_adjacent, 0), adjacent_lane.heading_at(0), random_speed)  # Vehicles in the slalom traffic cannot change lane, to foster the idea of a "slalom"
                self.road.vehicles.append(vehicle_adjacent)

            if counter >= max_rows:
                break
        
        return
        # spawn wall of vehicles
        for lane_index in range(num_lanes):
            lane = lanes[lane_index]
            x = lane.position(ego_position[0], ego_position[1])[0] + distance + counter * distance  # spawn behind the ego vehicle
            x += self.np_random.uniform(-2.5, 2.5)  # Add some randomness to the position of vehicles in the slalom traffic to make it more dynamic and less predictable
            vehicle = Vehicle(self.road, lane.position(x, 0), lane.heading_at(0), speed)  
            self.road.vehicles.append(vehicle)
            


    def _slalom_traffic(self):
        num_lanes = self.config["lanes_count"]
        lanes = self.road.network.lanes_list()

        num_vehicles = 8
        ego_position = self.vehicle.position
        distance = 30
        speed = self.np_random.uniform(*self.config["others_speed_range"])  # Random speed for vehicles in the slalom traffic to create more diverse traffic patterns
        counter = 0

        for i in range(num_vehicles // 4):
            lane_indices = self.np_random.permutation(num_lanes)

            # create random permutation of lane indices for this batch of vehicles
            for lane_index in lane_indices:
                # spawn a vehicle in the current lane
                lane = lanes[lane_index]
                x = lane.position(ego_position[0], ego_position[1])[0]  # spawn behind the ego vehicle
                x += distance + counter * distance  # space out vehicles by a certain distance
                x += self.np_random.uniform(-5, 5)

                counter += 1
                vehicle = Vehicle(self.road, lane.position(x, 0), lane.heading_at(0), speed)
                self.road.vehicles.append(vehicle)


    def _random_traffic(self):
        num_vehicles = 20
    
        for _ in range(num_vehicles):       
            speed = self.np_random.uniform(*self.config["others_speed_range"])  # Random speed for vehicles in the random traffic to create more diverse traffic patterns
            vehicle = IDMVehicle.create_random(
                self.road, 
                spacing=1 / self.config["vehicles_density"],
                speed=speed
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
