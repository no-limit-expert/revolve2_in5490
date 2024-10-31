import numpy as np

from genotypes.brain_genotype_simple import BrainGenotype as BrainGenotypeSimple

class BrainGenotype(BrainGenotypeSimple):

    @classmethod
    def initialize_brain(cls, rng) -> 'BrainGenotype':
        return BrainGenotype(brain={})

    def update_brain_parameters(self, brain_uuids, rng):
        for brain_uuid in brain_uuids:
            if brain_uuid not in self.brain.keys():
                self.brain[brain_uuid] = np.array(rng.random(2))
