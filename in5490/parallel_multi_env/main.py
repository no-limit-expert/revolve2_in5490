"""Main script for the example."""

import logging
import numpy as np
import multineat
import config
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from evaluator import Evaluator
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

from crossover import CrossoverReproducer
from selector import NSGAIISurvivorSelector, ParentSelector
# Ege sourced bayes_opt
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern

import statistics

import concurrent.futures

import pprint

def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")
    
    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # Generate a seed
    rng_seed = seed_from_time() % 2**32  # Ensure the seed is within the valid range

    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()
    
    # Intialize the evaluator that will be used to evaluate robots.
    environments = config.ENVS
    evaluators = [
        Evaluator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
            terrain=env
        ) for env in environments
    ]

    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = NSGAIISurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    # Create an initial population, as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes: list[Genotype] = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=0.0)
            for genotype in initial_genotypes
        ]
    )
     
    for env_n in range(len(environments)):
            logging.info(f"Evaluating population on env: {env_n+1} / {len(environments)}")
            population.individuals = learn_population(population.individuals, rng_seed, evaluators[env_n], env_n)  
        

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    save_to_db(dbengine, generation)

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")

    # Loop for every environment
    for i in range(config.NUM_BODY_GENERATIONS):
        logging.info(f"Generation: {i+1} / {config.NUM_BODY_GENERATIONS}.")

        # Reproduction. Get offspring
        parents, _ = parent_selector.select(population)
        offspring = crossover_reproducer.reproduce(parents, parent_population=population)

        # Train offspring and evaluate
        for env_n in range(len(environments)):
            logging.info(f"Evaluating offspring on env: {env_n+1} / {len(environments)}")
            offspring = learn_population(offspring, rng_seed, evaluators[env_n], env_n)
        
        
        offspring_genotypes = []
        offspring_fitnesses = []
        
        for c in offspring:
            offspring_genotypes.append(c.genotype)
            offspring_fitnesses.append(c.fitnesses)
        # Select offspring
        population, _ = survivor_selector.select(population=population, offspring=offspring_genotypes, offspring_fitness=offspring_fitnesses)
        
        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index+1,
            population=population,
        )

        # Finally save logs
        save_to_db(dbengine=dbengine, generation=generation)


def train_brain(individual: Individual, rng_seed: int, evaluator: Evaluator, env_n: int):
    # Find all active hinges in the body
    body = individual.genotype.develop_body()
    active_hinges = body.find_modules_of_type(ActiveHinge)
    # Make sure to initialize lists of params and fitnesses
    if individual.genotype.parameters_env is None:
        individual.genotype.parameters_env = [[] for _ in range(len(config.ENVS))]
    if individual.genotype.fitnesses_env is None:
        individual.genotype.fitnesses_env = [[] for _ in range(len(config.ENVS))]
    if individual.fitnesses == None:
        individual.fitnesses = [0.0]*3
    
    individual.genotype.parameters_env[env_n] = []
    individual.genotype.fitnesses_env[env_n] = []
        
    # If no hinges, skip
    if len(active_hinges) == 0:
        individual.fitness = 0.0
        individual.genotype.fitnesses_env[env_n].append(0.0)
        return individual

    # Create a structure for the CPG network from these hinges.
    # This also returns a mapping between active hinges and the index of there corresponding cpg in the network.
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    # Notify Bayesian Optimization (BO) of the bounds for the parameters
    pbounds = {}
    for i in range(cpg_network_structure.num_connections):
        pbounds[str(i)] = (-1, 1)
    
    # Initialize BO
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng_seed),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=config.KAPPA)
    )
    optimizer.set_gp_params(alpha=config.NOISE_ALPHA, 
                            kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE,length_scale_bounds="fixed"))

    for generation_index in range(config.NUM_BRAIN_GENERATIONS):
        logging.info(f"Training brain - iteration: {generation_index + 1} / {config.NUM_BRAIN_GENERATIONS}.")

        # Get the sampled solutions(parameters) from BO.
        next_point = optimizer.suggest()
        next_point = dict(sorted(next_point.items()))
        # Evaluate them.
        fitnesses = evaluator.evaluate([next_point.values()], body, cpg_network_structure, output_mapping)

        # Tell BO the fitnesses.
        optimizer.register(params=next_point, target=fitnesses[0])

        # TODO: These values are the brain parameters and fitness to save
        brain_parameters = list(next_point.values())
        fitness = fitnesses[0]

        individual.genotype.parameters_env[env_n].append(brain_parameters)
        individual.genotype.fitnesses_env[env_n].append(fitness)

    # Store best fitness in individual.fitness
    individual.fitnesses[env_n] = individual.genotype.fitnesses_env[env_n][-1]

    individual.fitness = penalized_geometric_mean(individual.fitnesses) # Column not used for anything 

    return individual

def learn_population(individuals: list[Individual], rng_seed: int, evaluator: Evaluator, env_n: int) -> list[Individual]:
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = [
            executor.submit(train_brain, individual, rng_seed, evaluator, env_n)
            for individual in individuals
        ]
    return [future.result() for future in futures]

def penalized_geometric_mean(fitnesses: list[float], alpha: float = 0.5) -> float:
    """
    Calculate the penalized geometric mean of fitness values, properly handling negatives.
    Negative fitness values yield worse scores than positive ones.
    
    Args:
        fitnesses: List of fitness values
        alpha: Penalty factor between 0 and 1 (default 0.5)
            - alpha = 0: No variance penalty
            - alpha = 1: Maximum variance penalty
    
    Returns:
        float: Penalized geometric mean value
    """
    import numpy as np
    
    if len(fitnesses) == 0:
        raise ValueError("Fitness list cannot be empty")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    values = np.array(fitnesses)
    
    # Find the shift needed to make all values positive for geometric mean
    min_value = min(min(values), 0)  # Get the minimum or 0, whichever is smaller
    shift = abs(min_value) + 1  # Add 1 to avoid log(0)
    
    # Calculate geometric mean by:
    # 1. Shift values to positive
    # 2. Take geometric mean
    # 3. Shift back to original scale
    shifted_values = values + shift
    geometric_mean = np.exp(np.mean(np.log(shifted_values))) - shift
    
    # Calculate coefficient of variation using original values
    mean = np.mean(values)
    if mean == 0:
        coefficient_of_variation = 0
    else:
        coefficient_of_variation = np.std(values) / abs(mean)
    
    # Apply variance penalty
    penalized_mean = geometric_mean * (1 - alpha * coefficient_of_variation)
    
    return float(penalized_mean)

def save_to_db(dbengine: Engine, generation: Generation) -> None:
    """
    Save the current generation to the database.

    :param dbengine: The database engine.
    :param generation: The current generation.
    """
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()


def main() -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OVERWITE_IF_EXISTS
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
