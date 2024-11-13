"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2, spider_v2
from revolve2.standards import terrains

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 8
INITIAL_STD = 0.5

NUM_BRAIN_GENERATIONS = 1
# BODY = spider_v2()

POPULATION_SIZE = 4
OFFSPRING_SIZE = 2
NUM_BODY_GENERATIONS = 2

KAPPA = 3
NU = 2.5
LENGTH_SCALE = 0.2
NOISE_ALPHA = 0

ENVS = [terrains.flat(), terrains.hills(length=10, height=0.25, num_edges= 50), terrains.steps(height=0.25)]