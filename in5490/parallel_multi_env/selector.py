from typing import Any

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

from revolve2.experimentation.evolution.abstract_elements import Selector
from revolve2.experimentation.optimization.ea import population_management, selection

import numpy as np
import numpy.typing as npt
from typing import Any, List, Tuple



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

class NSGAIISurvivorSelector(Selector):
    """Selector class for NSGA-II survivor selection."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize the NSGA-II selector.

        Args:
            rng: Random number generator
        """
        self.rng = rng

    def _calculate_dominance(self, fitnesses1: List[float], fitnesses2: List[float]) -> int:
        """Calculate dominance relationship between two solutions.
        
        Returns:
            -1 if fitnesses1 dominates fitnesses2
             1 if fitnesses2 dominates fitnesses1
             0 if neither dominates
        """
        if all(f1 <= f2 for f1, f2 in zip(fitnesses1, fitnesses2)) and \
           any(f1 < f2 for f1, f2 in zip(fitnesses1, fitnesses2)):
            return -1
        if all(f2 <= f1 for f1, f2 in zip(fitnesses1, fitnesses2)) and \
           any(f2 < f1 for f1, f2 in zip(fitnesses1, fitnesses2)):
            return 1
        return 0

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """Perform fast non-dominated sorting to identify Pareto fronts.
        
        Returns:
            List of fronts, where each front is a list of indices
        """
        n = len(population)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        # Calculate domination for each solution
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if either individual has no fitnesses
                if not population[i].fitnesses or not population[j].fitnesses:
                    continue

                print(population[i].fitnesses)
                print(population[j].fitnesses)
                    
                dom = self._calculate_dominance(
                    population[i].fitnesses,
                    population[j].fitnesses
                )
                if dom == -1:  # i dominates j
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dom == 1:  # j dominates i
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Find first front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Find subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _calculate_crowding_distance(self, individuals: List[Individual]) -> np.ndarray:
        """Calculate crowding distance for solutions in a front.
        
        Args:
            individuals: List of individuals in the front
            
        Returns:
            Array of crowding distances
        """
        n_solutions = len(individuals)
        if n_solutions <= 2:
            return np.full(n_solutions, np.inf)

        # Convert list of fitnesses to numpy array for easier manipulation
        fitnesses = np.array([ind.fitnesses for ind in individuals])
        n_objectives = fitnesses.shape[1]
        
        distances = np.zeros(n_solutions)
        
        for obj in range(n_objectives):
            # Sort solutions by current objective
            sorted_indices = np.argsort(fitnesses[:, obj])
            sorted_fitness = fitnesses[sorted_indices, obj]

            # Set boundary points to infinity
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Calculate distances
            obj_range = sorted_fitness[-1] - sorted_fitness[0]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distances[sorted_indices[i]] += (
                        sorted_fitness[i + 1] - sorted_fitness[i - 1]
                    ) / obj_range

        return distances

    def select(
        self,
        population: Population,
        offspring: list[Genotype],
        offspring_fitness: npt.NDArray[np.float_]
    ) -> tuple[Population, dict[str, Any]]:
        """Select survivors using NSGA-II selection.

        Args:
            population: Current population
            offspring: List of offspring genotypes
            offspring_fitness: Array of offspring fitness values

        Returns:
            New population and empty dictionary
        """
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance'."
            )

        # Create offspring individuals
        offspring_individuals = [
            Individual(
                genotype=g,
                fitness=0.0, # Unused
                fitnesses=f,
            )
            for g, f in zip(offspring, offspring_fitness)
        ]

        # Combine parent and offspring populations
        combined_pop = population.individuals + offspring_individuals

        # Get Pareto fronts
        fronts = self._fast_non_dominated_sort(combined_pop)

        # Select new population
        new_population = []
        front_idx = 0
        
        while len(new_population) + len(fronts[front_idx]) <= len(population.individuals):
            # Add whole front
            for idx in fronts[front_idx]:
                new_population.append(combined_pop[idx])
            front_idx += 1

        if len(new_population) < len(population.individuals):
            # Need to select partial front using crowding distance
            last_front = [combined_pop[i] for i in fronts[front_idx]]
            
            # Calculate crowding distances
            distances = self._calculate_crowding_distance(last_front)
            
            # Sort by crowding distance
            sorted_indices = np.argsort(-distances)  # Negative for descending order
            
            # Add solutions until we reach desired population size
            remaining = len(population.individuals) - len(new_population)
            for idx in sorted_indices[:remaining]:
                new_population.append(last_front[idx])

        return Population(individuals=new_population), {}