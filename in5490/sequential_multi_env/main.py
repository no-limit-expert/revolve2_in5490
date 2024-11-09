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


from revolve2.standards import terrains

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
    environments = [terrains.flat(), terrains.rugged_heightmap((20.0, 20.0), (2, 2)),]
    evaluators = [
        Evaluator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
            terrain=env
        ) for env in environments
    ]


    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng_seed)
    survivor_selector = SurvivorSelector(rng=rng_seed)
    crossover_reproducer = CrossoverReproducer(
        rng=rng_seed, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    mod_rob_evos = [
        ModularRobotEvolution(
            parent_selection=parent_selector,
            survivor_selection=survivor_selector,
            evaluator=eval,
            reproducer=crossover_reproducer,
        ) for eval in evaluators
    ]

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
    initial_bodies = [
        
    ]
    initial_fitnesses = evaluators[0].evaluate(initial_genotypes, )

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
            )
        ]
    )

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    save_to_db(dbengine, generation)

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")

    # Loop for every environment
    for env_n in range(len(environments)):
        logging.info(f"Done with environment {env_n + 1} / {len(environments)}.")
        # Loop same amount of generations for every environment
        while generation.generation_index < config.NUM_BODY_GENERATIONS:
            logging.info(f"Environment: {env_n}\tGeneration: {generation.generation_index + 1} / {config.NUM_BODY_GENERATIONS}.")
            # Train brain for every individual? Then decide fitness.
            for individual in population.individuals:
                # Find all active hinges in the body
                # active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
                active_hinges = individual.genotype.develop_body().find_modules_of_type(ActiveHinge)
                # If no hinges, skip
                if len(active_hinges) == 0:
                    continue

                # Create a structure for the CPG network from these hinges.
                # This also returns a mapping between active hinges and the index of there corresponding cpg in the network.
                (
                    cpg_network_structure,
                    output_mapping,
                ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

                # Initial parameter values for the brain.
                initial_mean = cpg_network_structure.num_connections * [0.5]

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
                    fitnesses = evaluators[env_n].evaluate([next_point.values()], 
                                                           individual.genotype.body, 
                                                           cpg_network_structure, 
                                                           output_mapping)

                    # Tell BO the fitnesses.
                    optimizer.register(params=next_point, target=fitnesses[0])

                    # TODO: These values are the brain parameters and fitness to save
                    brain_parameters = list(next_point.values())
                    fitness = fitnesses[0]

                    individual.genotype.parameters.append(brain_parameters)
                    individual.genotype.fitnesses.append(fitness)

                # Store best fitness in individual.fitness
                individual.fitness = individual.genotype.fitnesses[-1]

            #selection and reproduction
            population = mod_rob_evos[env_n].step(population)
            
            # Make it all into a generation and save it to the database.
            generation = Generation(
                experiment=experiment,
                generation_index=(),
                population=population,
            )

            # Finally save logs
            save_to_db(dbengine=dbengine, generation=generation)
        logging.info(f"Finished training on environment {env_n + 1}.")
        
# Could remove load from run_experiment
def train_brain():
    return

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
