from __future__ import annotations

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
        self._create_vehicles()

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
        self._simulate(action)
        self._refresh_traffic_around_controlled()

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
                self.config["lanes_count"], speed_limit=30
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
                speed=20.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            target_count = int(max(float(self.config["vehicles_density"]), 1.) * self.config["lanes_count"] * 200 / 80)
            target_count = min(target_count, others)

            self._spawn_vehicles_around_controlled(
                controlled_vehicle=vehicle,
                count=target_count,
                vehicle_type=other_vehicles_type,
                around_distance_ahead=150,
                around_distance_frac=0.5,
                min_gap_to_controlled=40,
                spawn_next_to_controlled=True
            )

    def _spawn_vehicles_around_controlled(
        self,
        controlled_vehicle: ControlledVehicle,
        count: int,
        vehicle_type: type[Vehicle],
        max_attempts_per_vehicle: int = 5,
        around_distance_ahead: float = 150,
        around_distance_frac: float = 0.7,
        min_gap_to_controlled: float = 40,
        spawn_next_to_controlled: bool = True
    ) -> None:
        """Spawn uncontrolled vehicles around a controlled vehicle with density-aware spacing."""
        if count <= 0:
            return

        initial_speed = self.np_random.uniform(17., 23.)
        
        lanes_count = self.config["lanes_count"]

        lane_from, lane_to, _ = controlled_vehicle.lane_index
        ego_longitudinal = controlled_vehicle.lane.local_coordinates(
            controlled_vehicle.position
        )[0]
        lane_length = controlled_vehicle.lane.length
     
        density = max(float(self.config["vehicles_density"]), 1.)
        default_spacing = 12 + 1.0 * initial_speed
        offset = (
            default_spacing
            * np.exp(-5 / 40 * len(self.road.network.graph[lane_from][lane_to]))
        )

        def is_available(candidate_lane: int, candidate_longitudinal: float) -> bool:
            for vehicle in self.road.vehicles:
                if vehicle.lane_index != (lane_from, lane_to, candidate_lane):
                    continue
                vehicle_longitudinal = vehicle.lane.local_coordinates(vehicle.position)[0]
                if abs(vehicle_longitudinal - candidate_longitudinal) < offset:
                    return False
            
            return True

        for _ in range(count):
            created = False
            for _ in range(max_attempts_per_vehicle):
                lane_id = int(self.np_random.integers(lanes_count))


                candidate_longitudinal = None
                if spawn_next_to_controlled and lane_id != controlled_vehicle.lane_index[2]:
                    candidate_longitudinal = ego_longitudinal + self.np_random.uniform(
                        -around_distance_ahead * around_distance_frac, around_distance_ahead
                    )
                else:
                    spawn_ahead = self.np_random.choice([True, False])
                    if spawn_ahead:
                        candidate_longitudinal = ego_longitudinal + self.np_random.uniform(
                            min_gap_to_controlled, around_distance_ahead
                        )
                    else:   
                        candidate_longitudinal = ego_longitudinal - self.np_random.uniform(
                            min_gap_to_controlled, around_distance_ahead * around_distance_frac
                        )

                candidate_longitudinal = float(
                    np.clip(candidate_longitudinal, 0.0, lane_length - 1.0)
                )

                if not is_available(lane_id, candidate_longitudinal):
                    continue

                vehicle = vehicle_type.make_on_lane(
                    self.road,
                    (lane_from, lane_to, lane_id),
                    longitudinal=candidate_longitudinal,
                    speed=initial_speed
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                created = True
                break

            if not created:
                vehicle = vehicle_type.create_random(self.road, spacing=offset)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _refresh_traffic_around_controlled(self) -> None:
        """
        Prune distant uncontrolled vehicles and respawn traffic around controlled vehicle.
        Assumes one controlled vehicle.
        """
        if not self.controlled_vehicles:
            return

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        prune_distance = 150
        # Determine target number of vehicles around each controlled vehicle based on density configuration
        density = max(float(self.config["vehicles_density"]), 1.)
        # 1 vehicle per lane and 40 m
        target_count = int(density * self.config["lanes_count"] * (2 * prune_distance) / 80)

        def is_near_any_controlled(vehicle: Vehicle) -> bool:
            return any(
                np.linalg.norm(vehicle.position - controlled_vehicle.position)
                <= prune_distance
                for controlled_vehicle in self.controlled_vehicles
            )

        controlled_vehicle = self.controlled_vehicles[0]

        # Prune vehicles that are too far from all controlled vehicles
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles or is_near_any_controlled(vehicle)
        ]

        nearby_count = sum(
            1
            for vehicle in self.road.vehicles
            if vehicle not in self.controlled_vehicles
            and np.linalg.norm(vehicle.position - controlled_vehicle.position)
            <= prune_distance
        )
        self._spawn_vehicles_around_controlled(
            controlled_vehicle=controlled_vehicle,
            count=target_count - nearby_count,
            vehicle_type=other_vehicles_type,
            around_distance_ahead=200,
            around_distance_frac=0.7,
            min_gap_to_controlled=100,
            spawn_next_to_controlled=False
        )

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
