import numpy as np
import torch
import torch.nn.functional as f


def perform_inference_in_plaintext(inference_task):
    samples = inference_task.query_samples.as_tensors

    name_to_ancestry = {}
    for i, sample in enumerate(samples):
        name, snps_1, snps_2 = sample

        print(f"Processing sample in plaintext '{name}' ({i / len(samples) * 100:.2f} %)")

        ancestry_1 = _infer_ancestry(snps_1, inference_task).numpy()
        ancestry_2 = _infer_ancestry(snps_2, inference_task).numpy()

        name_to_ancestry[name] = (ancestry_1, ancestry_2)

    return name_to_ancestry


def _infer_ancestry(snps, inference_task):
    # ------------ Compute the rest ------------

    per_population_scores = compute_per_population_scores(snps, inference_task)
    ancestry = compute_ancestry(per_population_scores, inference_task)

    return ancestry


def compute_per_population_scores(snps, inference_task):
    # ------------ Define a few utility variables ------------

    snp_count = inference_task.reference_panel.snp_count
    window_size = inference_task.model_parameters.window_size

    # If the number of SNPs is not divisible by the window size we simply ignore the remainder.
    # This shortcut simplifies computations wile not sacrificing much accuracy as the window
    # size (in the hundreds) is much smaller than the number of SNPs (typically in the millions).
    window_count = snp_count // window_size

    reference_panel = inference_task.reference_panel
    reference_panel_samples = inference_task.reference_panel.samples_as_tensor
    population_count = inference_task.reference_panel.population_count

    # ------------ Compute SNP matches ------------
    # 1 means a match, -1 means a mismatch at that position
    snps = snps * 2 - 1
    snp_matches = snps * reference_panel_samples  # The reference panel is already 1, -1 encoded.

    # ------------ Compute window similarity scores ------------
    # This reduces the length of the vector by the window size

    snp_matches = snp_matches.to(torch.float32)
    snp_matches = snp_matches.unsqueeze(1)

    averaging_kernel = torch.ones(1, 1, window_size).float() / window_size
    window_similarity_scores = f.conv1d(snp_matches, averaging_kernel, stride=window_size)

    # ------------ Find the highest similarity score for each population at each window ------------

    per_population_scores = torch.zeros((population_count, window_count))

    for current in range(population_count):
        lower, upper = reference_panel.get_population_bounds(current)
        current_scores = window_similarity_scores[lower:upper]
        current_max_scores, _ = torch.topk(current_scores, k=1, dim=0)
        per_population_scores[current] = current_max_scores

    return per_population_scores


def compute_ancestry(per_population_scores, inference_task):
    # ------------ Define a few utility variables ------------

    snp_count = inference_task.reference_panel.snp_count
    window_size = inference_task.model_parameters.window_size
    window_count = snp_count // window_size
    window_remainder = snp_count % window_size
    population_count = inference_task.reference_panel.population_count

    # ------------ Smooth out on the per population scores ------------

    per_population_scores = per_population_scores.reshape(population_count, 1, window_count)

    smoother_kernel = inference_task.model_parameters.smoother_weights_as_tensor
    smoother_kernel_size = len(smoother_kernel)
    smoother_kernel = smoother_kernel.reshape(1, 1, smoother_kernel_size)

    smooth_scores = f.conv1d(per_population_scores, smoother_kernel, padding=smoother_kernel_size // 2)

    # ------------ Find the population with the highest score at each window ------------

    _, best_populations = torch.topk(smooth_scores, k=1, dim=0)

    # ------------ Upsample each window to its original size ------------

    best_populations = best_populations.reshape(window_count)

    ancestry_upsampled = np.repeat(best_populations, window_size)

    # Pad the remainder with random populations to avoid bias for any of them
    remainder_padding = torch.randint(0, population_count, (window_remainder,))
    ancestry_upsampled = torch.cat((ancestry_upsampled, remainder_padding))

    return ancestry_upsampled
