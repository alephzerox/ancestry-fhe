import argparse

import testing
from argumentparser import parse_arguments, Action
from data.inferencetask import InferenceTask
from data.querymapping import QueryMapping
from inference_fhe import perform_inference_with_fhe, RunFhe
from utils.stopwatch import Stopwatch


def _perform_inference(
        model_parameters_path,
        reference_panel_samples_path,
        reference_panel_mapping_path,
        query_samples_path):
    #
    stopwatch = Stopwatch()

    inference_task = InferenceTask.load_from_files(
        model_parameters_path,
        reference_panel_samples_path,
        reference_panel_mapping_path,
        query_samples_path,
        stopwatch)

    execution_stopwatch = Stopwatch()

    perform_inference_with_fhe(
        inference_task,
        RunFhe(),
        10000,
        execution_stopwatch)


def _perform_tests(
        model_parameters_path,
        reference_panel_samples_path,
        reference_panel_mapping_path,
        test_samples_path,
        test_mapping_path):
    #
    stopwatch = Stopwatch()

    inference_task = InferenceTask.load_from_files(
        model_parameters_path,
        reference_panel_samples_path,
        reference_panel_mapping_path,
        test_samples_path,
        stopwatch)

    stopwatch.start("Loading test mapping...")
    test_mapping = QueryMapping.load_from_file(test_mapping_path)
    stopwatch.stop()

    testing.perform_tests(inference_task, test_mapping)


if __name__ == '__main__':
    try:
        arguments = parse_arguments()

        if arguments.action == Action.HELP:
            exit(0)

        if arguments.action == Action.INFER:
            _perform_inference(
                arguments.model_parameters,
                arguments.reference_panel_samples,
                arguments.reference_panel_mapping,
                arguments.query_samples)

        if arguments.action == Action.TEST:
            _perform_tests(
                arguments.model_parameters,
                arguments.reference_panel_samples,
                arguments.reference_panel_mapping,
                arguments.test_samples,
                arguments.test_mapping)

    except argparse.ArgumentError as parsing_error:
        print(parsing_error.message)
