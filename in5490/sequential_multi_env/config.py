"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2, spider_v2
from revolve2.standards import terrains

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 8
NUM_PARALLEL_PROCESSES = 12
INITIAL_STD = 0.5

NUM_BRAIN_GENERATIONS = 10
# BODY = spider_v2()

POPULATION_SIZE = 10
OFFSPRING_SIZE = 5
NUM_BODY_GENERATIONS = 10

KAPPA = 3
NU = 2.5
LENGTH_SCALE = 0.2
NOISE_ALPHA = 0

ENVS = [terrains.flat(), terrains.crater((10, 10), 0.3, 0, 0.1), terrains.steps(height=0.25)]