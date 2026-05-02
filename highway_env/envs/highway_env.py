from __future__ import annotations
from turtle import speed

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road import lane
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
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
            self._create_ego_vehicle()
            self._spawn_vehicles(spawn_range=(20, 300))  # Spawn an initial grid of vehicles around the ego vehicle
            self._spawn_vehicles(spawn_range=(-200, 20))  # Spawn an initial grid of vehicles around the ego vehicle
        else:
            self._create_vehicles()  # Original method to create vehicles randomly on the road

        lateral_wind = self.config.get("lateral_wind", 0)
        longitudinal_wind = self.config.get("longitudinal_wind", 0)

        for vehicle in self.controlled_vehicles:
            if isinstance(vehicle, ControlledVehicle):
                vehicle.set_action_offsets(longitudinal_wind, lateral_wind)

        if self.config.get("brake_failure", False):
            for vehicle in self.controlled_vehicles:
                if isinstance(vehicle, MDPVehicle):
                    vehicle.break_brake()

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


        # EXPERIMENTAL: Prune vehicles, effectively only behind the ego vehicle, to improve performance.
        self._extract_and_prune(ahead_distance_range=(1000, 1400), behind_distance_range=(200, 500))  
        self._simulate(action)
        
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

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
        graph = self.road.network.graph
        lane_from = list(graph.keys())[0]
        lane_to = list(graph[lane_from].keys())[0]
        lanes = graph[lane_from][lane_to]
        speed_per_lane = np.linspace(*self.config.get("other_speed_range", [20, 30]), num=len(lanes))
        speed_per_lane = speed_per_lane[::-1]  # Higher speeds on the right lanes, lower speeds on the left lanes, to foster lane changing and realistic traffic patterns
        speed_variation = self.config.get("speed_variation", 1)  # Add some variability to the speed of vehicles in the same lane to foster lane changing and realistic traffic patterns

        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=self.config.get("ego_initial_speed", 20.0),
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):       
                lane_id = self.np_random.choice(len(lanes))  # Favor spawning more vehicles on the right lanes to foster lane changing and realistic traffic patterns
                lane_speed = speed_per_lane[lane_id]
                speed = self.np_random.uniform(lane_speed - speed_variation, lane_speed + speed_variation)  # Add some variability to the speed of vehicles in the same lane
                vehicle = other_vehicles_type.create_random(
                    self.road, 
                    spacing=1 / self.config["vehicles_density"],
                    speed=speed,
                    lane_from=lane_from,
                    lane_to=lane_to,
                    lane_id=lane_id
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

            for _ in range(5):
                lane_id = self.np_random.choice(len(lanes))  # Favor spawning more vehicles on the right lanes to foster lane changing and realistic traffic patterns
                lane_speed = speed_per_lane[lane_id]
                speed = self.np_random.uniform(lane_speed - speed_variation, lane_speed + speed_variation)  # Add some variability to the speed of vehicles in the same lane
                vehicle = other_vehicles_type.create_random_behind(
                    self.road, 
                    spacing=1 / self.config["vehicles_density"],
                    speed=speed,
                    lane_from=lane_from,
                    lane_to=lane_to,
                    lane_id=lane_id,
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _create_ego_vehicle(self) -> None:
        """Create the ego vehicle in its initial position."""
        ego_lane = self.config.get("initial_lane_id")
        ego_speed = self.config.get("ego_initial_speed", 20.0)

        vehicle = Vehicle.create_random(
            self.road,
            speed=ego_speed,
            lane_id=ego_lane,
            spacing=self.config["ego_spacing"],
        )
        
        ego = self.action_type.vehicle_class(
            self.road, vehicle.position, vehicle.heading, vehicle.speed
        )
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

    def _spawn_vehicles(self, spawn_range: tuple[float, float], existing_vehicles: list = []) -> None:
        """Spawn new vehicles ahead of the ego vehicle to maintain traffic density."""
        ego = self.controlled_vehicles[0]  # Assuming single ego vehicle for simplicity
        ego_x = ego.position[0]

        BASE_DISTANCE = 20  # Base distance between vehicles at density=1
        # 2. Grid Parameters
        # dist_x: longitudinal spacing (e.g., 25m)
        # vehicles_count: Total cars to spawn in the grid
        dist_x = BASE_DISTANCE  / self.config["vehicles_density"]  # Scale distance by density to maintain spacing at different densities
        num_lanes = self.config["lanes_count"]

        num_rows = int((spawn_range[1] - spawn_range[0]) // dist_x)

        target_num_vehicles = num_rows * (num_lanes - 1)  # We will leave one lane free per row for variability

        if existing_vehicles is not None and len(existing_vehicles) >= target_num_vehicles:
            return  # We already have enough vehicles ahead, no need to spawn more

        speed_range = self.config.get("other_speed_range", [20, 30])
        lane_objects = self.road.network.lanes_list()
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # leave one random lane per subrow free
        spawn_mask = np.zeros((num_lanes, num_rows), dtype=bool)
        # randomly decide which lane to leave free for each row to ensure variability in traffic patterns and prevent perfectly aligned vehicles across lanes, which can cause unrealistic traffic patterns and collisions in the simulator
        for row in range(num_rows):
            free_lane = self.np_random.choice([i for i in range(num_lanes)])
            spawn_mask[free_lane, row] = True  # Mark this lane as occupied for this row

        # extract existing vehicles by lane
        vehicles_per_lane = {lane_id: [] for lane_id in range(num_lanes)}
        for lane_id in range(num_lanes):
            lane_vehicles = [v for v in existing_vehicles if v.lane_index[2] == lane_id]
            vehicles_per_lane[lane_id] = lane_vehicles

        row_centers = ego_x + spawn_range[0] + np.arange(spawn_range[0] + dist_x / 2, spawn_range[1], dist_x)
        for (lane_id, lane_vehicles) in vehicles_per_lane.items():
            # clip existing vehicle to subdivision positions to avoid collisions with new vehicles
            for vehicle in lane_vehicles:
                dist_to_row_centers = np.abs(vehicle.position[0] - row_centers)
                
                for idx, dist in enumerate(dist_to_row_centers):
                    if dist < dist_x / 2.5:
                        spawn_mask[lane_id, idx] = True  # Mark this lane as occupied for this row
                
        x_positions = np.zeros((num_lanes, num_rows))  # For debugging, to track where vehicles are spawned
        speeds = np.zeros((num_lanes, num_rows))  # For debugging, to track the speeds of spawned vehicles
        low_speed, high_speed = speed_range
        low_speed += 1.
        high_speed -= 1.  # Add a margin to the speed range to avoid spawning vehicles at the exact min/max speeds, which can cause unrealistic traffic patterns and collisions in the simulator
        base_speeds = np.linspace(*speed_range, num=num_lanes)  # Create a range of base speeds across the rows to add variability
        for row in range(0, num_rows):
            # subdivide the row into num_lanes subrows
            x_center = ego_x + spawn_range[0] + dist_x / 2 + (row * dist_x)
            x_min = x_center - dist_x / 2.5
            x_max = x_center + dist_x / 2.5
            x_bases =  np.linspace(x_min, x_max, num_lanes)  # Base x positions for this row, one per lane
            self.np_random.shuffle(x_bases)  # Shuffle the base positions to avoid perfectly aligned vehicles across lanes, which can cause unrealistic traffic patterns and collisions in the simulator
            
            # add some noise to the x positions to avoid perfectly aligned vehicles, which can cause unrealistic traffic patterns and collisions in the simulator
            noise = dist_x / num_lanes  / 4
            x_bases += self.np_random.normal(0, noise, size=num_lanes)  # Add noise within a quarter of the distance between vehicles
            x_bases.clip(x_min, x_max)  # Ensure the noisy positions are still within the bounds of the row

            x_positions[:, row] = x_bases  # Store these positions for debugging and later use when spawning vehicles

            speeds[:, row] = base_speeds
            speeds[:, row] += self.np_random.normal(0, 2, size=num_lanes)  # Add some noise to the speed for variability
            speeds[:, row] = np.clip(speeds[:, row], *speed_range)  # Ensure speeds are within the specified range
            self.np_random.shuffle(speeds[:, row])  # Shuffle speeds to avoid perfectly aligned vehicles across lanes, which can cause unrealistic traffic patterns and collisions in the simulator

        for lane_id in range(num_lanes):
            for row in range(num_rows):
                
                if spawn_mask[lane_id, row]:  # This lane is left free for this row
                    continue

                x_pos = x_positions[lane_id, row]
                if row > 0:
                    prev_x_pos = x_positions[lane_id, row - 1]
                    if abs(x_pos - prev_x_pos) < dist_x / 2.:
                        continue
                
                skip = False
                for existing_vehicle in vehicles_per_lane[lane_id]:
                    if abs(existing_vehicle.position[0] - x_pos) < dist_x / 2.:
                        skip = True
                        break
                if skip:
                    continue
                            
                # Create the vehicle
                lane = lane_objects[lane_id]
                vehicle = other_vehicles_type(
                    self.road, 
                    lane.position(x_pos, 0), 
                    heading=lane.heading_at(x_pos), 
                    speed=speeds[lane_id, row],
                    target_speed=speeds[lane_id, row]
                )
                
                # Randomize IDM parameters (politeness, etc.)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _refresh_surrounding_traffic(self):
        """Refresh the surrounding traffic by pruning distant vehicles and spawning new ones ahead and behind."""
        vehicles_ahead, vehicles_behind = self._extract_and_prune(ahead_distance_range=(200, 400), behind_distance_range=(200, 300))

        self._spawn_vehicles(spawn_range=(250, 400), existing_vehicles=vehicles_ahead)  # Example: spawn vehicles ahead
        self._spawn_vehicles(spawn_range=[-200., -300.], existing_vehicles=vehicles_behind)  # Example: spawn 1 row of vehicles behind

    def _extract_and_prune(self, ahead_distance_range: tuple[float, float], behind_distance_range: tuple[float, float] = (200, 300.0)) -> tuple[list[Vehicle], list[Vehicle]]:
        # check if there are at least traffic_threshold vehicles ahead within the specified distance
        ego = self.controlled_vehicles[0]  # Assuming single ego vehicle for simplicity
        vehicles_ahead = []
        vehicles_behind = []
        
        prune = []

        for vehicle in self.road.vehicles:
            if vehicle is not ego:
                distance = vehicle.position[0] - ego.position[0]
                if distance > 0:
                    if distance < ahead_distance_range[1] and distance > ahead_distance_range[0]:
                        vehicles_ahead.append(vehicle)
                    elif distance > ahead_distance_range[1]:
                        prune.append(vehicle)
                else:
                    if abs(distance) > behind_distance_range[0] and abs(distance) < behind_distance_range[1]:
                        vehicles_behind.append(vehicle)
                    elif abs(distance) > behind_distance_range[1]:
                        prune.append(vehicle)  

                if vehicle.crashed:
                    prune.append(vehicle)

        for vehicle in prune:
            self.road.vehicles.remove(vehicle)

        return vehicles_ahead, vehicles_behind

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
