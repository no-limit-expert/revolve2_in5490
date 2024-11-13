"""Standard fitness functions for modular robots."""

import math
import numpy as np
from pyrr import Vector3

from revolve2.modular_robot_simulation import ModularRobotSimulationState

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotSimulationState, SceneSimulationState

def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    print(f"Beginning: {begin_position.x}x {begin_position.y}y {begin_position.z}z")
    print(f"Beginning: {end_position.x}x {end_position.y}y {end_position.z}z")
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )

def detect_outliers(
        states: list[SceneSimulationState], robot: ModularRobot
) -> list[float]:
    previous = Vector3([0, 0, 0])
    distances = []
    for scene_state in states:
        current = scene_state.get_modular_robot_simulation_state(robot).get_pose().position

        distance = math.sqrt(
            (previous.x - current.x) ** 2
            + (previous.y - current.y) ** 2
        )
        distances.append(distance)

        previous = current

    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define the bounds for outliers
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    # Detect outliers
    return [x for x in distances if x < lower_bound or x > upper_bound]

def forward_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position

    if end_position.z < -1:
        return 0.0

    return end_position.y - begin_position.y