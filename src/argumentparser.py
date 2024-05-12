import argparse
import os
from enum import Enum


class Action(Enum):
    INFER = 0
    TEST = 1
    HELP = 2


def parse_arguments():
    parser = _build_argument_parser()
    arguments = parser.parse_args()

    if arguments.action == Action.HELP:
        parser.print_help()
        return

    def ensure_present(value, argument_name):
        if not value:
            raise argparse.ArgumentError(None, f"""Missing required parameter {argument_name}.""")

    # We must introduce some redundancy due to the limitations of argparse.
    ensure_present(arguments.model_parameters, "--model-parameters")
    ensure_present(arguments.reference_panel_samples, "--reference-panel-samples")
    ensure_present(arguments.reference_panel_mapping, "--reference-panel-mapping")

    if arguments.action == Action.INFER:
        ensure_present(arguments.query_samples, "--query-samples")
    elif arguments.action == Action.TEST:
        ensure_present(arguments.test_samples, "--test-samples")
        ensure_present(arguments.test_mapping, "--test-mapping")
    else:
        raise argparse.ArgumentError(None, "Invalid main action. Choose one of --infer, --evaluate, or --help.")

    return arguments


def _build_argument_parser():
    script_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        usage=f"""

{script_name} --infer --model-parameters <file> --reference-panel-samples <file> --reference-panel-mapping <file> --query-samples <file>

for inference, or

{script_name} --evaluate --model-parameters <file> --reference-panel-samples <file> --reference-panel-mapping <file> --test-samples <file> --test-mapping <file>

for model evaluation.""",
        description="Infers ancestry for a set of chromosome 22 full genome human query samples.",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False)

    # ----- Required arguments -----
    common_group = parser.add_argument_group("Common arguments")

    common_group.add_argument("--help", "-h", action="store_const", dest="action", const=Action.HELP,
                              help="Show this message and exit.")

    common_group.add_argument("--model-parameters", "-p", metavar="<file>",
                              help="The pickle file that contains the model parameters.\n\n")

    common_group.add_argument("--reference-panel-samples", "-rs", metavar="<file>",
                              help="The VCF file that contains the reference panel samples.\n\n")

    common_group.add_argument("--reference-panel-mapping", "-rm", metavar="<file>",
                              help="""The CSV file that maps the reference panel samples to their haplogroup in the following format:

<sample name>, <haplogroup>
<sample name>, <haplogroup>
...


    """)

    # ----- Infer group -----
    infer_group = parser.add_argument_group("To perform inference")

    infer_group.add_argument("--infer", "-i", action="store_const", dest="action", const=Action.INFER)

    infer_group.add_argument("--query-samples", "-qs", metavar="<file>",
                             help="""The VCF file that contains the query samples, i. e. the samples for which ancestry should be inferred."
Must be the full diploid genome of chromosome 22.\n\n""")

    # ----- Test group -----
    test_group = parser.add_argument_group("To perform model evaluation")

    test_group.add_argument("--test", "-t", action="store_const", dest="action", const=Action.TEST)

    test_group.add_argument("--test-samples", "-ts", metavar="<file>",
                            help="""The VCF file that contains the test samples.
Unlike query samples these samples have known ancestries and can be used to evaluate the model's accuracy.
Must be the full diploid genome of chromosome 22.\n\n""")

    test_group.add_argument("--test-mapping", "-tm", metavar="<file>",
                            help="""The Python pickle file that contains the actual ancestral populations of each 
SNP position of each sample in the test samples file.

It contains a pickled list of triplets of the form
(<sample name>, <ancestral populations for homologous chromosome 1>, <ancestral populations for homologous chromosome 2>)

Each of the two ancestral population arrays contains the actual ancestral population number at each SNP position.
The population numbers are consistent with the ones in the reference panel mapping file.

""")

    # Patch argparse bug.
    def argument_parser_error(message):
        raise argparse.ArgumentError(None, message)

    def argument_parser_exit(status=0, message=None):
        if status != 0:
            raise argparse.ArgumentError(None, message)

    parser.error = argument_parser_error
    parser.exit = argument_parser_exit
    parser.exit()

    return parser
