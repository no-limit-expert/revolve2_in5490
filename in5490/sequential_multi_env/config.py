"""Configuration parameters for this example."""

from revolve2.standards import terrains

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 50
INITIAL_STD = 0.5

NUM_BRAIN_GENERATIONS = 30

POPULATION_SIZE = 150
OFFSPRING_SIZE = 50
NUM_BODY_GENERATIONS = 150

KAPPA = 3
NU = 2.5
LENGTH_SCALE = 0.2
NOISE_ALPHA = 1e-10

ENVS = [terrains.flat(), terrains.hills(length=10, height=0.25, num_edges= 50), terrains.steps(height=0.25)]