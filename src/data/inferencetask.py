from data.modelparameters import ModelParameters
from data.querysamples import QuerySamples
from data.referencepanel import ReferencePanel


class InferenceTask:

    @classmethod
    def load_from_files(
            cls,
            model_parameters_path,
            reference_panel_samples_path,
            reference_panel_mapping_path,
            test_samples_path,
            stopwatch):
        #
        stopwatch.start("Loading model parameters...")
        model = ModelParameters.load_from_pickle(model_parameters_path)
        stopwatch.stop()

        stopwatch.start("Loading reference panel...")
        reference_panel = ReferencePanel.load_from_files(reference_panel_samples_path, reference_panel_mapping_path)
        stopwatch.stop()

        stopwatch.start("Loading samples...")
        query_samples = QuerySamples.load_from_file(test_samples_path)
        stopwatch.stop()

        task = InferenceTask(model, reference_panel, query_samples)
        return task

    @classmethod
    def generate(
            cls,
            window_size,
            smoother_weights,
            snp_count,
            population_count,
            average_population_size,
            query_sample_count,
            generator):
        #
        model_parameters = ModelParameters(window_size, smoother_weights)
        reference_panel = ReferencePanel.generate(snp_count, population_count, average_population_size, generator)
        query_samples = QuerySamples.generate(snp_count, query_sample_count, generator)

        inference_task = InferenceTask(model_parameters, reference_panel, query_samples)

        return inference_task

    def __init__(self, model_parameters, reference_panel, query_samples):
        assert query_samples.snp_count == reference_panel.snp_count

        self._model_parameters = model_parameters
        self._reference_panel = reference_panel
        self._query_samples = query_samples

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def reference_panel(self):
        return self._reference_panel

    @property
    def query_samples(self):
        return self._query_samples
