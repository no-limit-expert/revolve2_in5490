"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2, spider_v2

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 8
INITIAL_STD = 0.5

NUM_BRAIN_GENERATIONS = 1
# BODY = spider_v2()

POPULATION_SIZE = 2
OFFSPRING_SIZE = 1
NUM_BODY_GENERATIONS = 5

KAPPA = 3
NU = 2.5
LENGTH_SCALE = 0.2
NOISE_ALPHA = 0