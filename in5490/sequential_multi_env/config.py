"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 8
INITIAL_STD = 0.5

NUM_BRAIN_GENERATIONS = 3
BODY = gecko_v2()

POPULATION_SIZE = 100
OFFSPRING_SIZE = 50
NUM_BODY_GENERATIONS = 100