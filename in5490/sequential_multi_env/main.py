"""Main script for the example."""

import logging
from typing import Any, List
import concurrent

import config
import multineat
import numpy as np
import numpy.typing as npt
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
# New
from database_components.learn_generation import LearnGeneration
from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.learn_population import LearnPopulation

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.standards import terrains



import concurrent.futures
import logging
import time
from argparse import ArgumentParser

from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import config
from revolve2.modular_robot.body.base import ActiveHinge



class ParentSelector(Selector):
    """Selector class for parent selection."""

    rng: np.random.Generator
    offspring_size: int

    def __init__(self, offspring_size: int, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param offspring_size: The offspring size.
        :param rng: The rng generator.
        """
        self.offspring_size = offspring_size
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int_], dict[str, Population]]:
        """
        Select the parents.

        :param population: The population of robots.
        :param kwargs: Other parameters.
        :return: The parent pairs.
        """
        return np.array(
            [
                selection.multiple_unique(
                    selection_size=2,
                    population=[
                        individual.genotype for individual in population.individuals
                    ],
                    fitnesses=[
                        individual.fitness for individual in population.individuals
                    ],
                    selection_function=lambda _, fitnesses: selection.tournament(
                        rng=self.rng, fitnesses=fitnesses, k=2
                    ),
                )
                for _ in range(self.offspring_size)
            ],
        ), {"parent_population": population}


class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param rng: The rng generator.
        """
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[Population, dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance'."
            )

        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population.individuals],
            old_fitnesses=[i.fitness for i in population.individuals],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n, genotypes, fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=fitnesses, k=2
                ),
            ),
        )

        return (
            Population(
                individuals=[
                    Individual(
                        genotype=population.individuals[i].genotype,
                        fitness=population.individuals[i].fitness,
                    )
                    for i in original_survivors
                ]
                + [
                    Individual(
                        genotype=offspring[i],
                        fitness=offspring_fitness[i],
                    )
                    for i in offspring_survivors
                ]
            ),
            {},
        )


class CrossoverReproducer(Reproducer):
    """A simple crossover reproducer using multineat."""

    rng: np.random.Generator
    innov_db_body: multineat.InnovationDatabase
    innov_db_brain: multineat.InnovationDatabase

    def __init__(
        self,
        rng: np.random.Generator,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
    ):
        """
        Initialize the reproducer.

        :param rng: The ranfom generator.
        :param innov_db_body: The innovation database for the body.
        :param innov_db_brain: The innovation database for the brain.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body
        self.innov_db_brain = innov_db_brain

    def reproduce(
        self, population: npt.NDArray[np.int_], **kwargs: Any
    ) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param population: The parent pairs.
        :param kwargs: Additional keyword arguments.
        :return: The genotypes of the children.
        :raises ValueError: If the parent population is not passed as a kwarg `parent_population`.
        """
        parent_population: Population | None = kwargs.get("parent_population")
        if parent_population is None:
            raise ValueError("No parent population given.")

        offspring_genotypes = [
            Genotype.crossover(
                parent_population.individuals[parent1_i].genotype,
                parent_population.individuals[parent2_i].genotype,
                self.rng,
            ).mutate(self.innov_db_body, self.innov_db_brain, self.rng)
            for parent1_i, parent2_i in population
        ]
        return offspring_genotypes

def latin_hypercube(n, k, rng: np.random.Generator):
    """
    Generate Latin Hypercube samples.

    Parameters:
    n (int): Number of samples.
    k (int): Number of dimensions.

    Returns:
    numpy.ndarray: Array of Latin Hypercube samples of shape (n, k).
    """
    # Generate random permutations for each dimension
    perms = [rng.permutation(n) for _ in range(k)]

    # Generate the samples
    samples = np.empty((n, k))

    for i in range(k):
        # Generate the intervals
        interval = np.linspace(0, 1, n+1)

        # Assign values from each interval to the samples
        for j in range(n):
            samples[perms[i][j], i] = rng.uniform(interval[j], interval[j+1])

    return samples


def learn_population(genotypes, evaluator, dbengine, rng):
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = [
            executor.submit(learn_genotype, genotype, evaluator, rng)
            for genotype in genotypes
        ]
    result_objective_values = []
    genotypes = []
    for future in futures:

        objective_value, learn_generations = future.result()
        result_objective_values.append(objective_value)
        genotypes.append(learn_generations[0].genotype)

        for learn_generation in learn_generations:
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(learn_generation)
                session.commit()
    return result_objective_values, genotypes

def learn_genotype(genotype, evaluator, rng):
    # We get the brain uuids from the developed body, because if it is too big we don't want to learn unused uuids
    developed_body = genotype.develop_body()
    brain_uuids = set()
    for active_hinge in developed_body.find_modules_of_type(ActiveHinge):
        brain_uuids.add(active_hinge.map_uuid)
    brain_uuids = list(brain_uuids)
    genotype.update_brain_parameters(brain_uuids, rng)

    if len(brain_uuids) == 0:
        empty_learn_genotype = LearnGenotype(brain=genotype.brain, body=genotype.body)
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=empty_learn_genotype, objective_value=0)
            ]
        )
        return 0, [LearnGeneration(
            genotype=genotype,
            generation_index=0,
            learn_population=population,
        )]

    pbounds = {}
    for key in brain_uuids:
        pbounds['amplitude_' + str(key)] = [0, 1]
        pbounds['phase_' + str(key)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2**32))
    )
    optimizer.set_gp_params(alpha=config.ALPHA, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds=(config.LENGTH_SCALE - 0.01, config.LENGTH_SCALE + 0.01)))
    utility = UtilityFunction(kind="ucb", kappa=config.KAPPA)

    best_objective_value = None
    best_learn_genotype = None
    learn_generations = []
    lhs = latin_hypercube(config.NUM_RANDOM_SAMPLES, 2 * len(brain_uuids), rng)
    best_point = {}
    for i in range(config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")
        if i < config.NUM_RANDOM_SAMPLES:
            if config.EVOLUTIONARY_SEARCH:
                next_point = {}
                for key in brain_uuids:
                    next_point['amplitude_' + str(key)] = genotype.brain[key][0]
                    next_point['phase_' + str(key)] = genotype.brain[key][1]
            else:
                j = 0
                next_point = {}
                for key in brain_uuids:
                    next_point['amplitude_' + str(key)] = lhs[i][j]
                    next_point['phase_' + str(key)] = lhs[i][j + 1]
                    j += 2
                next_point = dict(sorted(next_point.items()))
        else:
            next_point = optimizer.suggest(utility)
            next_point = dict(sorted(next_point.items()))
            next_best = utility.utility([list(next_point.values())], optimizer._gp, 0)
            for _ in range(10000):
                possible_point = {}
                for key in best_point.keys():
                    possible_point[key] = best_point[key] + np.random.normal(0, config.NEIGHBOUR_SCALE)
                possible_point = dict(sorted(possible_point.items()))

                utility_value = utility.utility([list(possible_point.values())], optimizer._gp, 0)
                if utility_value > next_best:
                    next_best = utility_value
                    next_point = possible_point

        new_learn_genotype = LearnGenotype(brain={}, body=genotype.body)
        for brain_uuid in brain_uuids:
            new_learn_genotype.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                ]
            )
        robot = new_learn_genotype.develop(developed_body)

        # Evaluate them.
        start_time = time.time()
        objective_value = evaluator.evaluate(robot)
        end_time = time.time()
        new_learn_genotype.execution_time = end_time - start_time

        if best_objective_value is None or objective_value >= best_objective_value:
            best_objective_value = objective_value
            best_learn_genotype = new_learn_genotype
            best_point = next_point

        optimizer.register(params=next_point, target=objective_value)

        # From the samples and fitnesses, create a population that we can save.
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=new_learn_genotype, objective_value=objective_value)
            ]
        )
        # Make it all into a generation and save it to the database.
        learn_generation = LearnGeneration(
            genotype=genotype,
            generation_index=i,
            learn_population=population,
        )
        learn_generations.append(learn_generation)

    if config.OVERWRITE_BRAIN_GENOTYPE:
        for key, value in best_learn_genotype.brain.items():
            genotype.brain[key] = value
        genotype.brain = {k: v for k, v in sorted(genotype.brain.items())}

    return best_objective_value, learn_generations


## EXPERIMENT


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # CPPN innovation databases.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    Here we initialize the components used for the evolutionary process.
    
    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    environments = [terrains.flat(), terrains.rugged_heightmap(),]
    evals = [Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS, terrain=env) for env in environments]
    parent_selector = ParentSelector(offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )
    # mod_rob_evos : List[ModularRobotEvolution] = [ # remove this and use your own in this func
    #     ModularRobotEvolution(
    #         parent_selection=parent_selector,
    #         survivor_selection=survivor_selector,
    #         evaluator=eval,
    #         reproducer=crossover_reproducer,
    #     ) for eval in evals
    # ]

    # Create an initial population, as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses = evals[0].evaluate(initial_genotypes)

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


    # Start the actual optimization process.
    logging.info("Start optimization process.")
    for env_n in range(len(environments)):
        while generation.generation_index < config.NUM_GENERATIONS/len(environments):
            logging.info(
                f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
            )
            # Choose the parents and create offspring
            parents = parent_selector.select(population)
            offspring = crossover_reproducer.reproduce(parents)

            # Evaluate the offspring
            offspring_fitnesses, offspring_genotypes = learn_population(offspring, evals[env_n], dbengine, rng)
            
            # Offspring population
            offspring_individuals = [Individual(gen, fit) for gen,fit in zip(offspring_genotypes, offspring_fitnesses)]
            offspring_population = Population(offspring_individuals)
            # New population by selection
            population = survivor_selector.select(population, offspring_population)

            # Make it all into a generation and save it to the database.
            generation = Generation(
                experiment=experiment,
                generation_index=generation.generation_index + 1,
                population=population,
            )
            save_to_db(dbengine, generation)
            
        post_env_fitness = None
        if env_n < len(environments):
            post_env_fitness = evals[env_n+1]


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


if __name__ == "__main__":
    main()
