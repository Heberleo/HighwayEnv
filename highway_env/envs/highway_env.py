from __future__ import annotations
from turtle import speed

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        if self.config.get("spawn_mode", "standard") == "grid":
            self._create_grid_of_vehicles()  # New method to create a grid of vehicles around the ego
        else:
            self._create_vehicles()  # Original method to create vehicles randomly on the road

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=self.config.get("speed_limit", 30)
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=self.config.get("ego_initial_speed", 20.0),
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            speed_range = self.config.get("other_speed_range", [20, 30])
            for _ in range(others):      
                speed = self.np_random.uniform(*speed_range)

                vehicle = other_vehicles_type.create_random(
                    self.road, 
                    spacing=1 / self.config["vehicles_density"],
                    speed=speed
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _create_grid_of_vehicles(self) -> None:
        """Create a grid of vehicles centered around the ego vehicle."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        BASE_DISTANCE = 15  # Base distance between vehicles at density=1. Adjust as needed.
        
        # 1. Spawn Ego Vehicle
        # We place it at a fixed x=30 to leave some room behind it
        ego_lane = self.config.get("initial_lane_id")
        ego_speed = self.config.get("ego_initial_speed", 20.0)

        vehicle = Vehicle.create_random(
            self.road,
            speed=ego_speed,
            lane_id=ego_lane,
            spacing=self.config["ego_spacing"],
        )
        ego_x = vehicle.position[0]
        
        # Use your create_at logic or direct instantiation
        ego = self.action_type.vehicle_class(
            self.road, vehicle.position, vehicle.heading, vehicle.speed
        )
        self.controlled_vehicles = []
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        # 2. Grid Parameters
        # dist_x: longitudinal spacing (e.g., 25m)
        # vehicles_count: Total cars to spawn in the grid
        dist_x = BASE_DISTANCE  / self.config["vehicles_density"]  # Scale distance by density to maintain spacing at different densities
        num_lanes = self.config["lanes_count"]
        
        # Define the range of the grid relative to ego (e.g., 2 rows behind, 4 ahead)
        front_to_behind_ratio = self.config.get("front_to_behind_ratio", 2)  # More density means more cars ahead than behind
        rows = self.config["vehicles_count"] // num_lanes 
        rows_behind = int(rows / (1 + front_to_behind_ratio))
        rows_ahead = rows - rows_behind
        
        speed_range = self.config.get("other_speed_range", [20, 25])
        
        lane_objects = self.road.network.lanes_list()
        x_pos_behind = 0

        x_positions = np.zeros((num_lanes, rows_behind + rows_ahead))  # For debugging, to track where vehicles are spawned
        # 3. Spawn the Grid
        for lane_id in range(num_lanes):
            for row in range(-rows_behind, rows_ahead):
                print(f"Attempting to spawn vehicle in lane {lane_id}, row {row} (relative to ego)")
                # Don't spawn on the ego vehicle's exact spot
                if lane_id == ego_lane and row == 0:
                    continue
                
                row_ = row + 1 if row >= 0 else row  # Shift rows ahead by 1 to leave space for ego at row=0
                x_row = ego_x + (row_ * dist_x)

                x_pos = 0.
                if row_ == 1:  # First row ahead of ego, ensure it's at least dist_x away
                    x_pos = x_row + (row * dist_x) + self.np_random.uniform(0, dist_x/2)
                elif row_ == -1:  # First row behind ego, ensure it's at least dist_x away
                    x_pos = x_row + (row * dist_x) - self.np_random.uniform(0, dist_x/2)
                else:  # For other rows, ensure they are spaced from the last spawned vehicle in this lane
                    x_pos = x_row + (row * dist_x) + self.np_random.uniform(-dist_x/2, dist_x/2)

                # guard ego vehicle from being too close to any other vehicle, especially at low densities
                if abs(x_pos - ego_x) < BASE_DISTANCE * self.config.get("ego_spacing", 1.0):  # Ensure at least ego spacing times density scaled distance from ego
                    continue  # Skip this position if it's too close to ego after adjustment

                if x_pos - x_pos_behind < BASE_DISTANCE: 
                    x_pos += BASE_DISTANCE  # Ensure a minimum distance from the last spawned vehicle in this lane 
 
                x_pos_behind = x_pos

                row_index = row + rows_behind  # Convert to 0-based index for storage
                x_positions[lane_id, row_index] = x_pos  # Track this spawn position for debugging

        # variate x positions per row to avoid traffic walls
        for row in range(rows):
            row_positions = x_positions[:, row]
            # if positions are too close, add some noise
            for i in range(1, len(row_positions)):
                if abs(row_positions[i] - row_positions[i-1]) < BASE_DISTANCE:
                    # move away from the previous vehicle by adding some noise, direction depends on whether it's 
                    offset_direction = 1 if row_positions[i] > row_positions[i-1] else -1
                    offset = 2 * BASE_DISTANCE * offset_direction
                    row_positions[i] += offset
                

            x_positions[:, row] = row_positions
            
        # check if any vehicles are too close now, delete them if so (this can happen at low densities where the initial randomization doesn't create enough spacing)
        # only need to check within the same lane since vehicles in different lanes can be close without colliding
        for lane_id in range(num_lanes):
            lane_positions = x_positions[lane_id, :]
            lane_positions.sort()  # Sort positions to check adjacent vehicles
            for i in range(1, len(lane_positions)):
                if abs(lane_positions[i] - lane_positions[i-1]) < BASE_DISTANCE:
                    # If two vehicles are too close, remove the second one by setting its position to 0 (which will be ignored when spawning)
                    lane_positions[i] = 0
            x_positions[lane_id, :] = lane_positions

        for lane_id in range(num_lanes):
            for row in range(rows):
                # Add slight noise to speed
                x_pos = x_positions[lane_id, row]
                if x_pos == 0:  # This position was marked for removal due to being too close to another vehicle
                    continue

                v_speed = self.np_random.uniform(*speed_range)

                # with a density based probability, randomly skip spawning a vehicle to create more realistic traffic patterns
                if self.np_random.uniform(0, 1) > self.config["vehicles_density"] + 0.5:
                    continue

                # Create the vehicle
                lane = lane_objects[lane_id]
                vehicle = other_vehicles_type(
                    self.road, 
                    lane.position(x_pos, 0), 
                    heading=lane.heading_at(x_pos), 
                    speed=v_speed,
                    target_speed=v_speed
                )
                
                # Randomize IDM parameters (politeness, etc.)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        
    def set_scenario(self, scenario_type: str) -> None:
        """
        Updates environment parameters for a specific traffic scenario.
        Note: Call this before .reset() for a clean transition.
        """
        if scenario_type == "fast_sparse":
            self.config.update({
                "vehicles_count": 20,
                "vehicles_density": 0.4,
                "ego_initial_speed": 20.0,
                "other_speed_range": [18, 24],
                "ego_spacing": 1.5,
                "speed_limit": 30,
                "front_to_behind_ratio": 4,
                "spawn_mode": "grid"
            })
            
        elif scenario_type == "slow_dense":
            self.config.update({
                "vehicles_count": 20,
                "vehicles_density": 0.8,
                "ego_spacing": 1.5,
                "ego_initial_speed": 10.0,
                "other_speed_range": [6, 12],
                "speed_limit": 15,
                "front_to_behind_ratio": 2,
                "spawn_mode": "grid"
            })
        
        print(f"Scenario switched to: {scenario_type}")

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
