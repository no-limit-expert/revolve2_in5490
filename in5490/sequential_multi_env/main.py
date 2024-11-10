"""Main script for the example."""

import logging
import numpy as np
import cma
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

from revolve2.experimentation.evolution import ModularRobotEvolution
from crossover import CrossoverReproducer
from selector import SurvivorSelector, ParentSelector
# Ege sourced bayes_opt
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern

import concurrent.futures

from revolve2.standards import terrains

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
    environments = [terrains.flat(), terrains.hills()]
    evaluators = [
        Evaluator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
            terrain=env
        ) for env in environments
    ]


    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
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
    initial_bodies = [g.develop_body() for g in initial_genotypes]
    # initial_fitnesses = evaluators[0].evaluate(initial_genotypes, initial_bodies)

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=0.0)
            for genotype in initial_genotypes
        ]
    )

    learn_population(population.individuals, rng_seed, evaluators[0])

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    save_to_db(dbengine, generation)

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")



    # Loop for every environment
    for env_n in range(len(environments)):
        logging.info(f"Environment: {env_n + 1} / {len(environments)}.")
        # Loop same amount of generations for every environment
        for _ in range(int(config.NUM_BODY_GENERATIONS/len(environments))):
            logging.info(f"Environment: {env_n+1}\tGeneration: {generation.generation_index + 1} / {config.NUM_BODY_GENERATIONS}.")
            # Train brain for every individual? Then decide fitness.
            # for individual in population.individuals:
            #     train_brain(individual,rng_seed, evaluators[env_n])
            population.individuals = learn_population(population.individuals, rng_seed, evaluators[env_n])

            # Reproduction. Get offspring
            parents, _ = parent_selector.select(population)
            offspring = crossover_reproducer.reproduce(parents, parent_population=population)
            # Train offspring and evaluate
            # for c in offspring:
            #     train_brain(c, rng_seed, evaluators[env_n])

            offspring = learn_population(offspring, rng_seed, evaluators[env_n])

            offspring_genotypes = []
            offspring_fitnesses = []
            for c in offspring:
                offspring_genotypes.append(c.genotype)
                offspring_fitnesses.append(c.fitness)
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
        logging.info(f"Finished training on environment {env_n + 1}.")


def train_brain(individual: Individual, rng_seed: int, evaluator: Evaluator):
    # Find all active hinges in the body
    body = individual.genotype.develop_body()
    active_hinges = body.find_modules_of_type(ActiveHinge)

    # If no hinges, skip
    if len(active_hinges) == 0:
        individual.fitness = 0
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
        
        if individual.genotype.parameters is None:
            individual.genotype.parameters = []
        if individual.genotype.fitnesses is None:
            individual.genotype.fitnesses = []
    
        individual.genotype.parameters.append(brain_parameters)
        individual.genotype.fitnesses.append(fitness)

    # Store best fitness in individual.fitness
    individual.fitness = individual.genotype.fitnesses[-1]
    return individual

def learn_population(individuals: list[Individual], rng_seed: int, evaluator: Evaluator) -> list[Individual]:
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = [
            executor.submit(train_brain, individual, rng_seed, evaluator)
            for individual in individuals
        ]
    ret = [future.result() for future in futures]
    return ret



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
