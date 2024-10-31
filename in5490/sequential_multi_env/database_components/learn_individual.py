"""Individual class."""

from dataclasses import dataclass

from sqlalchemy import orm

from database_components._base import Base
from database_components.learn_genotype import LearnGenotype

from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class LearnIndividual(
    Base, GenericIndividual[LearnGenotype], population_table="learn_population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "learn_individual"
    objective_value: orm.Mapped[float] = orm.mapped_column(nullable=False)
