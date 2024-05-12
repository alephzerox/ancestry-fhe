import numpy as np

from inference_fhe import perform_inference_with_fhe, RunFhe
from utils.stopwatch import Stopwatch, PrintMessage


def perform_inference(inference_task):
    stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)
    execution_stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    stopwatch.start("Computing ancestries...")
    name_to_ancestry = perform_inference_with_fhe(
        inference_task,
        RunFhe(),
        10000,
        execution_stopwatch)
    stopwatch.stop()

    print()
    print("Predicted ancestries:")
    print()

    population_count = inference_task.reference_panel.population_count
    snp_count = inference_task.reference_panel.snp_count

    for sample_name, ancestry in name_to_ancestry.items():
        ancestry_1, ancestry_2 = ancestry

        counts_1 = np.bincount(ancestry_1, minlength=population_count)
        counts_2 = np.bincount(ancestry_2, minlength=population_count)

        counts = counts_1 + counts_2
        frequencies = counts / snp_count / 2

        print(f"Sample '{sample_name}':")
        for population, frequency in enumerate(frequencies):
            population_name = inference_task.reference_panel.to_population_name(population)

            print(f"    {population_name}: {frequency * 100:.1f} %")

        print()

    pass
