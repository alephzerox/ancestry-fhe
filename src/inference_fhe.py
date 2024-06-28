import itertools
import operator
from abc import ABC, abstractmethod

import numpy as np
import torch
from concrete import fhe

from inference_plain import compute_ancestry
from utils.stopwatch import Stopwatch


def perform_inference_with_fhe(
        inference_task,
        execution_level,
        intended_batch_size,
        execution_stopwatch,
        sample_limit=None):
    #
    window_size = inference_task.model_parameters.window_size
    batch_size = (intended_batch_size // window_size) * window_size

    assert batch_size % window_size == 0 and batch_size > 0

    server = _Server(inference_task.model_parameters, inference_task.reference_panel, execution_level, batch_size)
    name_to_ancestry = _run_client(server, inference_task, execution_stopwatch, sample_limit)

    return name_to_ancestry


# The code below explicitly separates code that would run on the client
# from code that would run on the server in an actual client-server
# deployment scenario. Concrete's actual deployment solutions are not
# used as this separation is for clarity.

def _run_client(server, inference_task, execution_stopwatch, sample_limit):
    fhe_client = server.request_clientspecs()

    stopwatch = Stopwatch()
    stopwatch.start("Generating keys...")
    fhe_client.generate_keys()
    stopwatch.stop()

    execution_stopwatch.start("Running circuit...")

    samples = inference_task.query_samples
    name_to_ancestry = {}
    for i, sample in enumerate(inference_task.query_samples):
        if sample_limit is not None and i >= sample_limit:
            break

        name, snps_1, snps_2 = sample

        print(f"Processing sample in FHE '{name}' ({i / samples.count * 100:.2f} %)")

        ancestry_1 = _infer_ancestry(snps_1, inference_task, fhe_client, server)
        ancestry_2 = _infer_ancestry(snps_2, inference_task, fhe_client, server)

        name_to_ancestry[name] = (ancestry_1, ancestry_2)

    execution_stopwatch.stop()

    return name_to_ancestry


def _infer_ancestry(snps, inference_task, fhe_client, server):
    snps_encoded = snps * 2 - 1

    snp_count = inference_task.reference_panel.snp_count
    window_size = inference_task.model_parameters.window_size
    window_count = snp_count // window_size
    population_count = inference_task.reference_panel.population_count
    batches = fhe_client.batch_indexes

    per_population_scores = np.zeros((1, 1, population_count, window_count))

    batch_count = len(batches) - 1
    for batch in range(batch_count):
        lower = batches[batch]
        upper = batches[batch + 1]

        batch_snps = snps_encoded[lower:upper]

        if not fhe_client.execution_level.isSimulation():
            print(f"    Processing batch {batch} ({batch / batch_count / 2 * 100:.2f} %)")

        batch_encrypted = fhe_client.encrypt(batch_snps, batch)
        batch_scores_encrypted = server.request_run_circuit(batch_encrypted, batch)
        batch_scores = fhe_client.decrypt(batch_scores_encrypted, batch)

        current_window_count = batch_scores.shape[3]
        lower_window = lower // window_size

        for population in range(population_count):
            for window in range(current_window_count):
                per_population_scores[0, 0, population, lower_window + window] = batch_scores[0, 0, population, window]

    per_population_scores = torch.from_numpy(per_population_scores)
    per_population_scores = per_population_scores.to(torch.float32)

    ancestry = compute_ancestry(per_population_scores, inference_task)
    ancestry = ancestry.numpy()

    return ancestry


class _Server:
    def __init__(self, model_parameters, reference_panel, execution_level, batch_size):
        self._model_parameters = model_parameters
        self._reference_panel = reference_panel
        self._execution_level = execution_level

        self._prepare_reference_panel_samples()
        self._compute_batches(batch_size)
        self._compile_batch_circuits()

    def _prepare_reference_panel_samples(self):
        reference_panel = self._reference_panel

        population_count = reference_panel.population_count
        snp_count = reference_panel.snp_count

        self._one_reference_per_population = np.zeros((population_count, snp_count), dtype=int)

        for population in range(population_count):
            lower, upper = reference_panel.get_population_bounds(population)
            single_sample = reference_panel.samples[lower]
            self._one_reference_per_population[population] = single_sample

    def _compute_batches(self, batch_size):
        snp_count = self._reference_panel.snp_count
        window_size = self._model_parameters.window_size

        batch_sizes = [0]

        shift = 0
        while snp_count - shift >= window_size:
            remaining = snp_count - shift
            batch_size = batch_size if remaining >= batch_size else (remaining // window_size) * window_size
            batch_sizes.append(batch_size)
            shift += batch_size

        self._batch_indexes = list(itertools.accumulate(batch_sizes, operator.add))

    def _compile_batch_circuits(self):
        batch_indexes = self._batch_indexes

        self._batch_circuits = []
        self._batch_samples = []

        stopwatch = Stopwatch()
        stopwatch.start("Compiling batch circuits...")

        for batch in range(len(self._batch_indexes) - 1):
            lower = batch_indexes[batch]
            upper = batch_indexes[batch + 1]
            batch_size = upper - lower

            current_batch_samples = self._one_reference_per_population[:, lower:upper]
            self._batch_samples.append(current_batch_samples)

            input_set_size = 100
            input_snps = np.random.randint(0, 2, (input_set_size, batch_size))
            input_snps = input_snps * 2 - 1

            self._active_batch_samples = current_batch_samples

            compiler = fhe.Compiler(self._compute_per_population_scores, {"snps": "encrypted"})
            configuration = fhe.Configuration(global_p_error=0.01)
            circuit = compiler.compile(input_snps, configuration)

            self._batch_circuits.append(circuit)

        stopwatch.stop()

    # This is what actually runs in FHE
    def _compute_per_population_scores(self, snps):
        # ------------ Define a few utility variables ------------

        reference_panel = self._reference_panel

        snp_count = self._active_batch_samples.shape[1]
        window_size = self._model_parameters.window_size
        population_count = reference_panel.population_count

        # ------------ Compute SNP matches ------------

        samples_slice = self._active_batch_samples
        snp_matches = snps * samples_slice

        # ------------ Compute window similarity scores ------------

        snp_matches_reshaped = snp_matches.reshape(1, 1, population_count, snp_count)

        sum_kernel = np.array([[[[1] * window_size]]])
        window_similarity_scores = fhe.conv(snp_matches_reshaped, sum_kernel, strides=(1, window_size))

        return window_similarity_scores

    def request_clientspecs(self):
        fhe_client = _FheClient(self._batch_circuits, self._batch_indexes, self._execution_level)
        return fhe_client

    def request_run_circuit(self, data_encrypted, batch):
        self._active_batch_samples = self._batch_samples[batch]
        circuit = self._batch_circuits[batch]
        results_encrypted = self._execution_level.run(circuit, data_encrypted)
        return results_encrypted


class _FheClient:
    def __init__(self, batch_circuits, batch_indexes, execution_level):
        self._batch_circuits = batch_circuits
        self._batch_indexes = batch_indexes
        self._execution_level = execution_level

    @property
    def batch_indexes(self):
        return self._batch_indexes

    @property
    def execution_level(self):
        return self._execution_level

    def generate_keys(self):
        for i, circuit in enumerate(self._batch_circuits):
            self._execution_level.generate_keys(circuit)

    def encrypt(self, snps_plaintext, batch):
        circuit = self._batch_circuits[batch]
        data_encrypted = self._execution_level.encrypt(circuit, snps_plaintext)
        return data_encrypted

    def decrypt(self, data_encrypted, batch):
        circuit = self._batch_circuits[batch]
        data_plaintext = self._execution_level.decrypt(circuit, data_encrypted)
        return data_plaintext


class _ExecutionLevel(ABC):

    @abstractmethod
    def isSimulation(self):
        pass

    @abstractmethod
    def generate_keys(self, circuit):
        pass

    @abstractmethod
    def encrypt(self, circuit, snps_plaintext):
        pass

    @abstractmethod
    def decrypt(self, circuit, data_encrypted):
        pass

    @abstractmethod
    def run(self, circuit, data_encrypted):
        pass


class RunFhe(_ExecutionLevel):

    def isSimulation(self):
        return False

    def generate_keys(self, circuit):
        circuit.keygen()

    def encrypt(self, circuit, snps_plaintext):
        data_encrypted = circuit.encrypt(snps_plaintext)
        return data_encrypted

    def decrypt(self, circuit, data_encrypted):
        data_plaintext = circuit.decrypt(data_encrypted)
        return data_plaintext

    def run(self, circuit, data_encrypted):
        ret_val = circuit.run(data_encrypted)
        return ret_val


class SimulateFhe(_ExecutionLevel):

    def isSimulation(self):
        return True

    def generate_keys(self, circuit):
        pass

    def encrypt(self, circuit, snps_plaintext):
        return snps_plaintext

    def decrypt(self, circuit, data_encrypted):
        return data_encrypted

    def run(self, circuit, data_encrypted):
        ret_val = circuit.simulate(data_encrypted)
        return ret_val
