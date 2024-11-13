"""Evaluator class."""

import math

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)

from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)

from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        terrain: Terrain = terrains.flat()
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrain

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
        body: Body,
        cpg_network_structure: CpgNetworkStructure,
        output_mapping: list[tuple[int, ActiveHinge]]
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param solutions: Solutions to evaluate.
        :returns: Fitnesses of the solutions.
        """
        # Create robots from the brain parameters.
        robots = [
            ModularRobot(
                body=body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=output_mapping,
                ),
            )
            for params in solutions
        ]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the xy displacements.
        if len(fitness_functions.detect_outliers(scene_states[0], robot)) > 1:
            return [0.0]

        return [fitness_functions.forward_displacement(
                scene_states[0][0].get_modular_robot_simulation_state(robot),
                scene_states[0][-1].get_modular_robot_simulation_state(robot),
            )]