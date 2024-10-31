import copy
import math
import uuid
from pyrr import Vector3

import numpy as np
import sqlalchemy.orm as orm
import json
from sqlalchemy import event
from sqlalchemy.engine import Connection

from genotypes.body_genotype import BodyGenotype
from genotypes.brain_genotype_simple import BrainGenotype
import config
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1, BrickV1

class BodyDeveloper:
    queue: list
    grid: list
    body: BodyV1
    reverse_direction = {
        'left': 'right',
        'up': 'up',
        'front': 'front',
        'right': 'left',
        'down': 'down',
        'back': 'back',
        'attachment': 'attachment'
    }

    def __init__(self, initial_module):
        self.grid = [Vector3([0, 0, 0])]
        self.body = BodyV1()

        initial_module.body_module = self.body.core_v1

        self.queue = []
        self.queue.append((initial_module, initial_module.body_module, False))


    def develop(self):
        while len(self.queue) > 0:
            current_module, current_body_module, module_mirror = self.queue.pop(0)
            for directions, new_module in current_module.children.items():
                direction_mirror = module_mirror
                for direction in directions:
                    new_body_module = new_module.get_body_module(direction_mirror)
                    if module_mirror:
                        direction = self.reverse_direction[direction]
                    setattr(current_body_module, direction, new_body_module)

                    grid_position = self.body.grid_position(new_body_module)
                    if grid_position in self.grid or len(self.grid) > config.MAX_NUMBER_OF_MODULES:
                        setattr(current_body_module, direction, None)
                        continue

                    self.grid.append(grid_position)

                    self.queue.append((new_module, new_body_module, direction_mirror))
                    direction_mirror = not direction_mirror

            if isinstance(current_body_module, ActiveHingeV1):
                BodyDeveloper.develop_hinge(current_module, current_body_module, module_mirror)

    @staticmethod
    def develop_hinge(hinge_module, hinge_body_module, mirror):
        hinge_body_module.map_uuid = hinge_module.brain_index
        if mirror:
            hinge_body_module.reverse_phase = hinge_module.reverse_phase_value
        else:
            hinge_body_module.reverse_phase = 0


class ModuleGenotype:
    rotation: float
    temp_rotation: float
    children: dict
    possible_children: list
    type: str
    body_module = None
    reverse_rotation = {
        RightAngles.DEG_0.value: 0.0,
        RightAngles.DEG_90.value: RightAngles.DEG_270.value,
        RightAngles.DEG_180.value: RightAngles.DEG_180.value,
        RightAngles.DEG_270.value: RightAngles.DEG_90.value
    }
    reverse_direction = {
        'left': 'right',
        'up': 'up',
        'front': 'front',
        'right': 'left',
        'down': 'down',
        'back': 'back',
        'attachment': 'attachment'
    }

    def __init__(self, rotation):
        self.rotation = rotation
        self.children = {}

    def get_body_module(self, reverse):
        self.temp_rotation = self.rotation
        if reverse:
            self.temp_rotation = self.reverse_rotation[self.rotation]

    def add_random_module_to_connection(self, index: int, rng: np.random.Generator, brain: BrainGenotype):
        for directions in self.possible_children:
            if tuple(directions) in self.children.keys():
                index = self.children[tuple(directions)].add_random_module_to_connection(index, rng, brain)
            else:
                if index == 1:
                    self.children[tuple(directions)] = self.choose_random_module(rng, brain)
                index -= 1
            if index == 0:
                return 0
        return index

    def add_random_module_to_block(self, index: int, rng: np.random.Generator, brain: BrainGenotype):
        if index == 1:
            connection_choice = rng.choice(np.array(self.possible_children, dtype=object))
            module_to_add = self.choose_random_module(rng, brain)
            existing_module = None
            if tuple(connection_choice) in list(self.children.keys()):
                existing_module = self.children[tuple(connection_choice)]
            self.children[tuple(connection_choice)] = module_to_add

            if existing_module is not None:
                if isinstance(module_to_add, HingeGenotype):
                    module_to_add.children[tuple(['attachment'])] = existing_module
                if isinstance(module_to_add, BrickGenotype):
                    module_to_add.children[tuple(['front'])] = existing_module

            return 0
        for module in self.children.values():
            index = module.add_random_module_to_block(index - 1, rng, brain)
            if index == 0:
                return 0
        return index

    def get_amount_possible_connections(self):
        possible_connections = 0
        for directions in self.possible_children:
            if tuple(directions) in self.children.keys():
                possible_connections += self.children[tuple(directions)].get_amount_possible_connections()
            else:
                possible_connections += 1
        return possible_connections

    def get_amount_leaf_nodes(self):
        leaves = 0

        for module in self.children.values():
            leaves += module.get_amount_leaf_nodes()

        if leaves == 0:
            leaves = 1
        return leaves

    def get_amount_nodes(self):
        nodes = 1

        for module in self.children.values():
            nodes += module.get_amount_nodes()

        return nodes

    def get_amount_modules(self):
        nodes = 1

        for directions, module in self.children.items():
            nodes += module.get_amount_modules() * len(directions)

        return nodes

    def get_amount_hinges(self):
        nodes = 0

        for directions, module in self.children.items():
            nodes += module.get_amount_hinges() * len(directions)

        return nodes

    def is_leaf_node(self):
        return not bool(self.children)

    def remove_leaf_node(self, index):
        for direction, module in self.children.items():
            if module.is_leaf_node():
                if index == 1:
                    self.children.pop(direction)
                    return 0
                index -= 1
            else:
                index = module.remove_leaf_node(index)
                if index == 0:
                    return 0

        return index

    def remove_node(self, index):
        for direction, module in self.children.items():
            if index == 1:
                child = self.children.pop(direction)

                for child_direction in child.children.keys():

                    temp_key = list(child_direction)
                    if self.type == 'hinge' and list(child_direction) == ['front']:
                        temp_key = ['attachment']
                    elif self.type in ['core', 'brick'] and list(child_direction) == ['attachment']:
                        temp_key = list(direction)
                    elif self.type == 'core' and list(child_direction) in [['front'], ['back']]:
                        temp_key = ['front', 'back']

                    if temp_key in self.possible_children and tuple(temp_key) not in self.children.keys():
                        self.children[tuple(temp_key)] = child.children[child_direction]

                return 0
            index = module.remove_node(index - 1)
            if index == 0:
                return 0
        return index

    def check_for_brains(self):
        uuids = []
        for module in self.children.values():
            recursive_uuids = module.check_for_brains()
            for recursive_uuid in recursive_uuids:
                if recursive_uuid not in uuids:
                    uuids.append(recursive_uuid)
        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        for module in self.children.values():
            module.switch_brain(rng, brain)

    def reverse_phase_function(self, value):
        for module in self.children.values():
            module.reverse_phase_function(value)

    def serialize(self):
        serialized = {'type': self.type, 'rotation': self.rotation, 'children': {}}

        for directions, module in self.children.items():
            direction_string = ",".join(directions)
            serialized['children'][direction_string] = module.serialize()

        return serialized

    def deserialize(self, serialized):
        self.type = serialized['type']
        for direction, child in serialized['children'].items():
            if isinstance(direction, str):
                direction = tuple(map(str, direction.split(',')))
            if child['type'] == 'brick':
                child_object = BrickGenotype(child['rotation'])
            else:
                child_object = HingeGenotype(child['rotation'])
            self.children[direction] = child_object.deserialize(child)

        return self

    def choose_random_module(self, rng: np.random.Generator, brain: BrainGenotype):
        module_chooser = rng.random()
        rotation = rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value, RightAngles.DEG_270.value])

        if module_chooser < 0.5:
            module = BrickGenotype(rotation)
        else:
            module = HingeGenotype(rotation)

            new_brain_chooser = rng.random()
            if config.CONTROLLERS == -1 and new_brain_chooser < config.NEW_HINGE_NEW_BRAIN:
                module.brain_index = brain.add_new(rng)
            else:
                module.brain_index = rng.choice(list(brain.brain.keys()))

        return module


class CoreGenotype(ModuleGenotype):
    possible_children = [['left'], ['right'], ['front', 'back']]
    type = 'core'
    rotation = 0.0
    possible_phase_differences = [0, math.pi]
    reverse_phase = 0

    def get_amount_nodes(self):
        nodes = 0

        for module in self.children.values():
            nodes += module.get_amount_nodes()

        return nodes

    def serialize(self):
        serialized = super().serialize()

        phase_difference_to_value = {
            0: 0,
            0.5 * math.pi: 2,
            math.pi: 1,
            1.5 * math.pi: 3,
        }

        serialized['reverse_phase'] = phase_difference_to_value[self.reverse_phase]

        return serialized

    def deserialize(self, serialized):
        super().deserialize(serialized)

        value_to_phase_difference = {
            0: 0,
            2: 0.5 * math.pi,
            1: math.pi,
            3: 1.5 * math.pi,
        }

        self.reverse_phase = value_to_phase_difference[serialized['reverse_phase']]

        return self


class BrickGenotype(ModuleGenotype):
    possible_children = [['left'], ['right'], ['front'], ['up'], ['down']]
    type = 'brick'

    def get_body_module(self, reverse):
        super().get_body_module(reverse)
        self.body_module = BrickV1(self.temp_rotation)
        return self.body_module


class HingeGenotype(ModuleGenotype):
    possible_children = [['attachment']]
    brain_index = -1
    reverse_phase_value = 0
    type = 'hinge'

    def get_body_module(self, reverse):
        super().get_body_module(reverse)
        self.body_module = ActiveHingeV1(self.temp_rotation)
        return self.body_module

    def check_for_brains(self):
        uuids = super().check_for_brains()
        if self.brain_index not in uuids:
            uuids.append(self.brain_index)

        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        if rng.random() > config.SWITCH_BRAIN:
            self.brain_index = rng.choice(list(brain.brain.keys()))

        super().switch_brain(rng, brain)

    def reverse_phase_function(self, value):
        self.reverse_phase_value = value

        super().reverse_phase_function(value)

    def serialize(self):
        serialized = super().serialize()
        serialized['brain_index'] = str(self.brain_index)

        return serialized

    def deserialize(self, serialized):
        super().deserialize(serialized)
        self.brain_index = uuid.UUID(serialized['brain_index'])

        return self

    def get_amount_hinges(self):
        nodes = 1

        for module in self.children.values():
            nodes += module.get_amount_hinges()

        return nodes


class BodyGenotypeDirect(orm.MappedAsDataclass, BodyGenotype):
    """SQLAlchemy model for a direct encoding body genotype."""

    body: CoreGenotype

    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    def __init__(self, body: CoreGenotype):
        self.body = body

    @classmethod
    def initialize_body(cls, rng: np.random.Generator, brain: BrainGenotype):
        number_of_modules = rng.integers(config.INIT_MIN_MODULES, config.INIT_MAX_MODULES)
        body = CoreGenotype(0.0)
        current_number_of_modules = 0
        while current_number_of_modules < number_of_modules:
            amount_possible_connections = body.get_amount_possible_connections()
            connection_to_add = rng.integers(1, amount_possible_connections + 1)
            body.add_random_module_to_connection(connection_to_add, rng, brain)
            current_number_of_modules = body.get_amount_modules()

        body.reverse_phase = rng.choice(body.possible_phase_differences)

        return BodyGenotypeDirect(body)

    def mutate_body_start(self, rng: np.random.Generator, brain: BrainGenotype):
        body = None
        mutation_chooser = 0
        mutation_accepted = False
        while not mutation_accepted:
            body = copy.deepcopy(self.body)
            mutation_chooser = rng.random()

            if mutation_chooser < 0.45:
                for _ in range(rng.integers(1, config.MAX_ADD_MODULES + 1)):
                    amount_nodes = body.get_amount_nodes() + 1
                    node_to_add = rng.integers(1, amount_nodes + 1)
                    body.add_random_module_to_block(node_to_add, rng, brain)
                mutation_accepted = body.get_amount_modules() < config.MAX_NUMBER_OF_MODULES * 1.1
            elif mutation_chooser <= 0.9:
                for _ in range(rng.integers(1, config.MAX_DELETE_MODULES + 1)):
                    amount_nodes = body.get_amount_nodes()
                    if amount_nodes == 0:
                        break
                    node_to_remove = rng.integers(1, amount_nodes + 1)
                    body.remove_node(node_to_remove)

                if config.CONTROLLERS == -1:
                    used_brains = body.check_for_brains()
                    brain.remove_unused(used_brains, rng)
                mutation_accepted = True
            elif mutation_chooser <= 1:
                new_phase = body.reverse_phase
                while new_phase == body.reverse_phase:
                    new_phase = rng.choice(body.possible_phase_differences)
                body.reverse_phase = new_phase
                mutation_accepted = True
            else:
                body.switch_brain(rng, brain)

                if config.CONTROLLERS == -1:
                    used_brains = body.check_for_brains()
                    brain.remove_unused(used_brains, rng)
                mutation_accepted = False
        return BodyGenotypeDirect(body), mutation_chooser

    def get_brain_uuids(self):
        return self.body.check_for_brains()

    @staticmethod
    def crossover_body(parent1: 'BodyGenotypeDirect', parent2: 'BodyGenotypeDirect', rng: np.random.Generator):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if len(parent1.body.children) == 0 or len(parent2.body.children) == 0:
            return child1, child2

        parent_1_branch_chooser = rng.choice(list(parent1.body.children))
        parent_2_branch_chooser = rng.choice(list(parent2.body.children))

        child1.body.children[parent_1_branch_chooser] = copy.deepcopy(parent2.body.children[parent_2_branch_chooser])
        child2.body.children[parent_2_branch_chooser] = copy.deepcopy(parent1.body.children[parent_1_branch_chooser])

        child1.body.children[parent_1_branch_chooser].rotation = (
            rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value,
                        RightAngles.DEG_270.value]))
        child2.body.children[parent_2_branch_chooser].rotation = (
            rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value,
                        RightAngles.DEG_270.value]))

        return child1, child2


    def develop_body(self):
        if config.REVERSE_PHASE:
            self.body.reverse_phase_function(self.body.reverse_phase)

        body_developer = BodyDeveloper(self.body)
        body_developer.develop()
        return body_developer.body


@event.listens_for(BodyGenotypeDirect, "before_update", propagate=True)
@event.listens_for(BodyGenotypeDirect, "before_insert", propagate=True)
def _update_serialized_body(
        mapper: orm.Mapper[BodyGenotypeDirect],
        connection: Connection,
        target: BodyGenotypeDirect,
) -> None:
    target._serialized_body = str(target.body.serialize()).replace("'", "\"")


@event.listens_for(BodyGenotypeDirect, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeDirect, context: orm.QueryContext) -> None:
    serialized_dict = json.loads(target._serialized_body)
    target.body = CoreGenotype(0.0).deserialize(serialized_dict)
