import gzip
import pickle


# Maps to each nucleotide position its ancestral population.
# At each position we track the index of the ancestral group.
# This index can be found in the reference panel mapping file along with the name of the group.
# Used for validation and testing by comparing predicted populations to these actual values.
class QueryMapping:

    @classmethod
    def load_from_file(cls, path):
        with gzip.open(path, "rb") as mapping_file:
            mappings = pickle.load(mapping_file)

        mapping = QueryMapping(mappings)
        return mapping

    @classmethod
    def generate(cls, snp_count, sample_names, population_count, generator):
        mappings = []

        for i, name in enumerate(sample_names):
            indexes_1 = generator.integers(low=0, high=population_count, size=snp_count)
            indexes_2 = generator.integers(low=0, high=population_count, size=snp_count)
            mapping = (name, indexes_1, indexes_2)
            mappings.append(mapping)

        query_mapping = QueryMapping(mappings)
        return query_mapping

    def __init__(self, mappings):
        self._sample_to_ancestry = {}

        for i, sample in enumerate(mappings):
            name, snps_1, snps_2 = sample
            self._sample_to_ancestry[name] = (snps_1, snps_2)

    def get_ancestry(self, sample_name):
        ancestry = self._sample_to_ancestry[sample_name]
        return ancestry

    def __str__(self):
        ret_val = ""
        for name, ancestry in self._sample_to_ancestry.items():
            ancestry_1, ancestry_2 = ancestry
            ret_val += f"{name}:\n{ancestry_1}\n{ancestry_2}\n\n"

        return ret_val
