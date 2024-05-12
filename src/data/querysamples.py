import allel
import torch


class QuerySamples:

    @classmethod
    def load_from_file(cls, path):
        vcf = allel.read_vcf(path)

        diploid_samples = vcf["calldata/GT"].transpose(1, 2, 0)
        sample_names = vcf["samples"]

        test_samples = QuerySamples(diploid_samples, sample_names)
        return test_samples

    @classmethod
    def generate(cls, snp_count, sample_count, generator):
        samples = generator.integers(low=0, high=2, size=(sample_count, 2, snp_count))
        sample_names = [f"Sample {i}" for i in range(sample_count)]
        test_samples = QuerySamples(samples, sample_names)
        return test_samples

    def __init__(self, samples, sample_names):
        self.sample_count, _, _ = samples.shape

        assert self.sample_count > 0
        assert self.sample_count == len(sample_names)

        self._samples = []
        for i, sample in enumerate(samples):
            snps_1, snps_2 = sample[0], sample[1]
            name = sample_names[i]
            sample = (name, snps_1, snps_2)
            self._samples.append(sample)

        self._samples_as_tensors = None

        self._sample_names = sample_names

    @property
    def count(self):
        return self.sample_count

    @property
    def snp_count(self):
        ret_val = len(self._samples[0][1])
        return ret_val

    def __iter__(self):
        self._iteration_index = 0
        return self

    def __next__(self):
        if self._iteration_index < self.count:
            ret_val = self._samples[self._iteration_index]
            self._iteration_index += 1
            return ret_val
        else:
            raise StopIteration

    @property
    def as_tensors(self):
        if self._samples_as_tensors == None:
            self._samples_as_tensors = []
            for i, sample in enumerate(self._samples):
                name, snps_1, snps_2 = sample
                snps_1_as_tensor = torch.from_numpy(snps_1)
                snps_2_as_tensor = torch.from_numpy(snps_2)

                sample_as_tensors = (name, snps_1_as_tensor, snps_2_as_tensor)
                self._samples_as_tensors.append(sample_as_tensors)

        return self._samples_as_tensors

    def __str__(self):
        ret_val = ""
        for i, sample in enumerate(self):
            name, sample_1, sample_2 = sample
            ret_val += f"{name}:\n{sample_1}\n{sample_2}\n\n"

        return ret_val
