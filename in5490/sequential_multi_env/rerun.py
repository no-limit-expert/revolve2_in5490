"""Rerun the best robot between all experiments."""

import logging

import config
from database_components import Genotype, Individual
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.standards import fitness_functions, terrains

def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    with Session(dbengine) as ses:
        row = ses.execute(
            select(Genotype, Individual.fitness)
            .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
            .order_by(Individual.fitness.desc())
            .limit(1)
        ).one()
        assert row is not None

        genotype: Genotype = row[0]
        fitness = row[1]

    parameters = genotype.parameters[-1]

    logging.info(f"Best fitness: {fitness}")
    logging.info(f"Best parameters: {parameters}")

    # Prepare the body and brain structure
    body = genotype.develop_body()
    active_hinges = body.find_modules_of_type(ActiveHinge)
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    # Create the evaluator.
    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        # terrain=terrains.hills(length=7.5, height=0.1, num_edges= 75)
        # terrain=terrains.flat()
        terrain=terrains.crater((10, 10), 0.3, 0, 0.1)
        # terrain=terrains.steps(height=0.25)
        
    )

    # Show the robot.
    evaluator.evaluate([parameters], cpg_network_structure=cpg_network_structure, body=body, output_mapping=output_mapping)


if __name__ == "__main__":
    main()
