import numpy as np

from inference_fhe import perform_inference_with_fhe, RunFhe, SimulateFhe
from inference_plain import perform_inference_in_plaintext
from utils.stopwatch import Stopwatch, PrintMessage


def perform_tests(inference_task, query_mapping):
    inference_clear_seconds, accuracy_plaintext = None, None
    accuracy_fhe = None
    inference_fhe_full_seconds, inference_fhe_execution_seconds = None, None

    lower, upper = 0, inference_task.reference_panel.sample_count
    snp_count = inference_task.reference_panel.snp_count
    samples = inference_task.reference_panel.samples
    window_size = 200
    window_count = snp_count // window_size

    same_count = 0
    for window in range(window_count):
        same = True
        base_value = samples[lower][window * window_size : (window + 1) * window_size]
        for sample in range(lower + 1, upper):
            current_value = samples[sample][window * window_size : (window + 1) * window_size]

            snp_difference_count = np.sum(current_value != base_value)

            if snp_difference_count > 0:
                same = False
                break

        if same:
            same_count += 1

    performance_sample_count = 1

    inference_clear_seconds, accuracy_plaintext = _perform_clear_accuracy_test(inference_task, query_mapping)
    accuracy_fhe = _perform_fhe_accuracy_test(inference_task, query_mapping)
    inference_fhe_full_seconds, inference_fhe_execution_seconds = _perform_fhe_performance_test(inference_task, performance_sample_count)

    sample_count = inference_task.query_samples.count

    performance_clear_string = f"{inference_clear_seconds / sample_count:.2f}" if inference_clear_seconds is not None else "n/a"
    performance_fhe_with_setup_string = f"{inference_fhe_full_seconds / performance_sample_count:.2f}" if inference_fhe_full_seconds is not None else "n/a"
    performance_fhe_excluding_setup_string = f"{inference_fhe_execution_seconds / performance_sample_count:.2f}" if inference_fhe_execution_seconds is not None else "n/a"
    accuracy_clear_string = f"{accuracy_plaintext * 100:.1f}" if accuracy_plaintext is not None else "n/a"
    accuracy_fhe_string = f"{accuracy_fhe * 100:.1f}" if accuracy_fhe is not None else 'n/a'

    print(f"""
Summary
    Number of samples: {sample_count}

    Performance
        Clear:                   {performance_clear_string} s/sample
        FHE (including setup):   {performance_fhe_with_setup_string} s/sample
        FHE (excluding setup):   {performance_fhe_excluding_setup_string} s/sample

    Accuracy
        Clear: {accuracy_clear_string} %
        FHE:   {accuracy_fhe_string} %
""")


def _perform_clear_accuracy_test(inference_task, query_mapping):
    stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    print("----------------------------------------------------")
    stopwatch.start("Performing ancestry inference in plaintext...")

    prediction_plaintext = perform_inference_in_plaintext(inference_task)

    stopwatch.stop()

    inference_clear_seconds = stopwatch.elapsed_seconds
    accuracy_plaintext = _compute_accuracy(prediction_plaintext, query_mapping)

    return inference_clear_seconds, accuracy_plaintext


def _perform_fhe_accuracy_test(inference_task, query_mapping):
    stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)
    execution_stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    # Run the accuracy tests in simulation to save time
    print("----------------------------------------------------")
    stopwatch.start("Performing ancestry inference in FHE simulation...")

    prediction_fhe = perform_inference_with_fhe(
        inference_task,
        SimulateFhe(),
        10_000,
        execution_stopwatch)

    stopwatch.stop()

    accuracy_fhe = _compute_accuracy(prediction_fhe, query_mapping)

    return accuracy_fhe


def _perform_fhe_performance_test(inference_task, performance_sample_count):
    stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)
    execution_stopwatch = Stopwatch(PrintMessage.ON_START_AND_STOP)

    # Run one test in actual FHE to measure performance
    print("----------------------------------------------------")
    stopwatch.start("Performing ancestry inference in FHE...")

    perform_inference_with_fhe(
        inference_task,
        RunFhe(),
        10_000,
        execution_stopwatch,
        performance_sample_count)

    stopwatch.stop()

    inference_fhe_full_seconds = stopwatch.elapsed_seconds
    inference_fhe_execution_seconds = execution_stopwatch.elapsed_seconds

    return inference_fhe_full_seconds, inference_fhe_execution_seconds


def _compute_accuracy(prediction, mapping):
    item_count = len(prediction.items())
    if item_count == 0:
        return 0

    accuracy_sum = 0
    for name, predicted_ancestries in prediction.items():
        predicted_1, predicted_2 = predicted_ancestries
        actual_1, actual_2 = mapping.get_ancestry(name)

        accuracy_1 = _get_similarity(predicted_1, actual_1)
        accuracy_2 = _get_similarity(predicted_2, actual_2)

        accuracy_sum += accuracy_1 + accuracy_2

    accuracy = accuracy_sum / item_count / 2
    return accuracy


def _get_similarity(left, right):
    length = len(left)
    match_count = np.count_nonzero(left == right)
    match_ratio = match_count / length
    return match_ratio

