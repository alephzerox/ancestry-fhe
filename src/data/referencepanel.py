import csv
import itertools
import operator

import allel
import numpy
import torch


class ReferencePanel:

    @classmethod
    def load_from_files(cls, samples_path, mapping_path):
        vcf = allel.read_vcf(samples_path)
        sample_names = vcf["samples"]

        samples = vcf["calldata/GT"].transpose(1, 2, 0)

        sample_to_population = {}
        population_to_index = {}
        with (open(mapping_path, "r") as mapping_file):
            csv_reader = csv.reader(mapping_file, delimiter=',')
            next(csv_reader, None)  # Skip header

            for row in csv_reader:
                sample_name, population_name, population_index = row
                population_index = int(population_index)

                sample_to_population[sample_name] = population_name

                # The population name <-> index mapping is redundant in the CSV file
                # as it's there in each line but creating a different mapping file
                # would have complicated the command line.
                #
                # Having a well-defined mapping is important to make sure that
                # the indexes in the test mapping are consistent with the ones
                # in the reference panel mapping.
                assert not population_name in population_to_index.keys() \
                       or population_to_index[population_name] == population_index

                population_to_index[population_name] = population_index

        # Population numbers are used as indexes, so they must start at 0
        assert 0 in population_to_index.values()

        sample_index_to_population = []
        for i, sample_name in enumerate(sample_names):
            population_name = sample_to_population[sample_name]
            sample_index_to_population.append(population_name)

        reference_panel = ReferencePanel(samples, sample_index_to_population, population_to_index)
        return reference_panel

    @classmethod
    def generate(
            cls,
            snp_count,
            population_count,
            average_population_size,
            generator):
        #
        sample_count = population_count * average_population_size
        samples = generator.integers(low=0, high=2, size=(sample_count, 2, snp_count))

        population_name_to_index = {f"Population {i}": i for i in range(population_count)}
        population_names = list(population_name_to_index.keys())
        sample_population_names = population_names.copy()  # Guarantee at least one sample per population
        indexes_rest = generator.integers(low=0, high=population_count, size=sample_count - population_count).tolist()
        sample_population_names_rest = [population_names[i] for i in indexes_rest]
        sample_population_names = sample_population_names + sample_population_names_rest

        panel = ReferencePanel(samples, sample_population_names, population_name_to_index)
        return panel

    def __init__(self, samples, sample_to_population_name, population_to_index):
        self._population_to_index = population_to_index
        sample_count, ploidy, sample_length = samples.shape

        assert sample_count > 0
        assert len(sample_to_population_name) == sample_count

        self._population_to_name = {}
        for name, index in population_to_index.items():
            assert not index in self._population_to_name.keys()
            self._population_to_name[index] = name

        haploid_samples = samples.reshape((sample_count * ploidy, sample_length))
        haploid_samples = haploid_samples * 2 - 1  # 1, -1 encode in advance

        # Sort the samples so that the samples that belong to the same population
        # are grouped together. This will speed up certain computations.

        population_sample_counts = [0] * self.population_count
        population_indexes = []
        for i, sample in enumerate(haploid_samples):
            population_name = sample_to_population_name[i // 2]
            population_index = self.to_population_index(population_name)
            population_indexes.append(population_index)
            population_sample_counts[population_index] += 1

        self._population_start_indexes = list(itertools.accumulate(population_sample_counts, operator.add))
        self._population_start_indexes.insert(0, 0)

        population_current_indexes = self._population_start_indexes.copy()

        sorted_samples = numpy.zeros(haploid_samples.shape, dtype=int)
        for i, sample in enumerate(haploid_samples):
            population_index = population_indexes[i]
            sample_index = population_current_indexes[population_index]
            population_current_indexes[population_index] += 1
            sorted_samples[sample_index] = sample

        self._samples = sorted_samples
        self._samples_as_tensor = None

    @property
    def snp_count(self):
        ret_val = self._samples[0].shape[0]
        return ret_val

    @property
    def population_names(self):
        ret_val = self._population_to_index.keys()
        return ret_val

    @property
    def population_count(self):
        ret_val = len(self.population_names)
        return ret_val

    def to_population_name(self, index):
        ret_val = self._population_to_name[index]
        return ret_val

    def to_population_index(self, name):
        ret_val = self._population_to_index[name]
        return ret_val

    @property
    def samples(self):
        return self._samples

    @property
    def samples_as_tensor(self):
        if self._samples_as_tensor == None:
            self._samples_as_tensor = torch.from_numpy(self._samples)

        return self._samples_as_tensor

    def get_population_bounds(self, population):
        ret_val = (
            self._population_start_indexes[population],
            self._population_start_indexes[population + 1])

        return ret_val

    def __str__(self):
        ret_val = ""
        for population in range(self.population_count):
            lower, upper = self.get_population_bounds(population)
            samples = self._samples[lower:upper]
            ret_val += f"Population {population} ({len(samples)} samples):\n{samples}\n\n"

        return ret_val
