import logging
import os

import numpy as np
import torch


def _component_vae_loss(component, data):
    prepared, spike_input = component._to_spike_input(data)
    x_recon, q_z, p_z, _ = component.vae(spike_input, scheduled=True)
    loss = component.vae.loss_function_mmd(prepared, x_recon, q_z, p_z)
    return loss["loss"].mean().item()


def test_components(components, test_loaders, device, args):
    """Evaluate all experts by routing each batch to the lowest-ELBO VAE."""
    del args
    os.makedirs("recon_results", exist_ok=True)

    total_correct = 0
    total_samples = 0

    for component in components:
        component.to(device)
        component.eval()

    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders, start=1):
            task_correct = 0
            task_samples = 0

            for data, labels in test_loader:
                labels = labels.to(device)
                elbos = [_component_vae_loss(component, data) for component in components]
                best_component = components[int(np.argmin(elbos))]

                predictions = best_component.test_classifier(data).to(device)
                correct = int((predictions == labels).sum().item())
                batch_size = int(labels.size(0))

                task_correct += correct
                task_samples += batch_size
                total_correct += correct
                total_samples += batch_size

            accuracy = task_correct / task_samples if task_samples > 0 else 0.0
            logging.info("Task %s accuracy: %.2f%%", task_id, accuracy * 100)
            print(f"Task {task_id} accuracy: {accuracy * 100:.2f}%")

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    logging.info("Overall accuracy: %.2f%%", overall_accuracy * 100)
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")

    return overall_accuracy
