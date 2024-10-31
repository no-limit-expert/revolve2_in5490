"""Main script for the example."""

import logging

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



from revolve2.standards import terrains

def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")


    

    # Create an initial population, as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng_seed,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(initial_genotypes)

    

    # Create an rng seed.
    rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    

    # Intialize the evaluator that will be used to evaluate robots.
    environments = [terrains.flat(), terrains.rugged_heightmap((20.0, 20.0), (2, 2)),]
    evaluators = [
        Evaluator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
            cpg_network_structure=cpg_network_structure,
            body=config.BODY,
            output_mapping=output_mapping,
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
    for env_n in environments:
        while ...:
             # Find all active hinges in the body
            active_hinges = config.BODY.find_modules_of_type(ActiveHinge)

            # Create a structure for the CPG network from these hinges.
            # This also returns a mapping between active hinges and the index of there corresponding cpg in the network.
            (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

            # Intialize the evaluator that will be used to evaluate robots.
            evaluator = Evaluator(
                headless=True,
                num_simulators=config.NUM_SIMULATORS,
                cpg_network_structure=cpg_network_structure,
                body=config.BODY,
                output_mapping=output_mapping,
            )

            # Initial parameter values for the brain.
            initial_mean = cpg_network_structure.num_connections * [0.5]

            # Initialize the cma optimizer.
            options = cma.CMAOptions()
            options.set("bounds", [-1.0, 1.0])
            options.set("seed", rng_seed)
            opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)

            while opt.countiter < config.NUM_BRAIN_GENERATIONS:
                logging.info(f"Generation {opt.countiter + 1} / {config.NUM_GENERATIONS}.")

                # Get the sampled solutions(parameters) from cma.
                solutions = opt.ask()

                # Evaluate them.
                fitnesses = evaluators[0].evaluate(solutions)

                # Tell cma the fitnesses.
                # Provide them negated, as cma minimizes but we want to maximize.
                opt.tell(solutions, -fitnesses)

                # From the samples and fitnesses, create a population that we can save.
                population = Population(
                    individuals=[
                        Individual(genotype=Genotype(parameters), fitness=fitness)
                        for parameters, fitness in zip(solutions, fitnesses)
                    ]
                )

                # Make it all into a generation and save it to the database.
                generation = Generation(
                    experiment=experiment,
                    generation_index=opt.countiter,
                    population=population,
                )
                
            logging.info("Saving generation.")
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(generation)
                session.commit()


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
