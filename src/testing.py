import numpy as np

from inference_fhe import perform_inference_with_fhe, RunFhe, SimulateFhe
from inference_plain import perform_inference_in_plaintext
from utils.stopwatch import Stopwatch, PrintMessage


def perform_tests(inference_task, query_mapping):
    #
    stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    print("----------------------------------------------------")
    stopwatch.start("Performing ancestry inference in plaintext...")
    prediction_plaintext = perform_inference_in_plaintext(inference_task)
    stopwatch.stop()

    inference_clear_seconds = stopwatch.elapsed_seconds

    execution_stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    # Run the accuracy tests in simulation to save time
    print("----------------------------------------------------")
    stopwatch.start("Performing ancestry inference in FHE simulation...")

    prediction_fhe = perform_inference_with_fhe(
        inference_task,
        SimulateFhe(),
        100000,
        execution_stopwatch)

    stopwatch.stop()
    print("----------------------------------------------------")
    print()

    # Run one test in actual FHE to measure performance
    stopwatch.start("Performing ancestry inference in FHE...")

    performance_sample_count = 1
    perform_inference_with_fhe(
        inference_task,
        RunFhe(),
        50000,
        execution_stopwatch,
        performance_sample_count)

    stopwatch.stop()
    print("----------------------------------------------------")

    inference_fhe_full_seconds = stopwatch.elapsed_seconds
    inference_fhe_execution_seconds = execution_stopwatch.elapsed_seconds

    sample_count = inference_task.query_samples.count

    accuracy_plaintext = _compute_accuracy(prediction_plaintext, query_mapping)
    accuracy_fhe = _compute_accuracy(prediction_fhe, query_mapping)

    print(f"""
Summary
    Number of samples: {sample_count}

    Performance
        Clear:                   {inference_clear_seconds / sample_count:.2f} s/sample
        FHE (including setup):   {inference_fhe_full_seconds / performance_sample_count:.2f} s/sample
        FHE (excluding setup):   {inference_fhe_execution_seconds / performance_sample_count:.2f} s/sample
            
    Accuracy
        Clear: {accuracy_plaintext * 100:.1f} %
        FHE:   {accuracy_fhe * 100:.1f} %
""")


def _compute_accuracy(prediction, mapping):
    accuracy_sum = 0
    for name, predicted_ancestries in prediction.items():
        predicted_1, predicted_2 = predicted_ancestries
        actual_1, actual_2 = mapping.get_ancestry(name)

        accuracy_1 = _get_similarity(predicted_1, actual_1)
        accuracy_2 = _get_similarity(predicted_2, actual_2)

        accuracy_sum += accuracy_1 + accuracy_2

    accuracy = accuracy_sum / len(prediction.items()) / 2
    return accuracy


def _get_similarity(left, right):
    length = len(left)
    match_count = np.count_nonzero(left == right)
    match_ratio = match_count / length
    return match_ratio
