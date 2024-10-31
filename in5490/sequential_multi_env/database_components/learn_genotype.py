"""Genotype class."""

from __future__ import annotations

from database_components._base import Base
# from genotypes.brain_genotype_simple import BrainGenotype
from genotypes.brain_genotype_cppn_simple import BrainGenotype as BrainGenotypeCppn
# from genotypes.body_genotype_direct import BodyGenotypeDirect
from genotypes.body_genotype_cppn import BodyGenotypeCppn

from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot

import sqlalchemy.orm as orm


class LearnGenotype(Base, HasId, BrainGenotypeCppn, BodyGenotypeCppn):
    """A genotype that is an array of parameters."""

    __tablename__ = "learn_genotype"
    execution_time: orm.Mapped[float] = orm.mapped_column(default=0.0)

    def develop(self, body) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)
