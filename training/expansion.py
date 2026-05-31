import logging

import torch


def _flatten_features(features):
    return features.float().reshape(features.size(0), -1)


def _generated_samples(component, num_samples):
    samples, _ = component.vae.get_sample(num_samples)
    return samples


def _encode_generated_features(gen_component, encode_component, num_samples):
    samples = _generated_samples(gen_component, num_samples)
    return encode_component.encode_mmd_features(samples)


def _encode_data_features(component, data):
    return component.encode_mmd_features(data)


def compute_pairwise_distance(comp1, comp2, num_samples=64, n_steps=8):
    """Return the symmetric latent MSE between two experts."""
    del n_steps
    logging.info("Computing pairwise expert distance")

    z1_self = _flatten_features(_encode_generated_features(comp1, comp1, num_samples)).mean(dim=0)
    z1_cross = _flatten_features(_encode_generated_features(comp1, comp2, num_samples)).mean(dim=0)
    z2_self = _flatten_features(_encode_generated_features(comp2, comp2, num_samples)).mean(dim=0)
    z2_cross = _flatten_features(_encode_generated_features(comp2, comp1, num_samples)).mean(dim=0)

    mse_loss = torch.nn.MSELoss()
    mse1 = mse_loss(z1_self, z1_cross).item()
    mse2 = mse_loss(z2_self, z2_cross).item()
    return (mse1 + mse2) / 2


def cross_encode(gen_component, encode_component, num_samples, n_steps):
    """Generate from one expert and encode with another expert."""
    del n_steps
    features = _encode_generated_features(gen_component, encode_component, num_samples)
    return _flatten_features(features).mean(dim=0)


def get_sample_z(gen_component, encode_component, num_samples, n_steps):
    """Generate samples and return their architecture-native MMD features."""
    del n_steps
    return _encode_generated_features(gen_component, encode_component, num_samples)


def get_sample_z_from_data(encode_component, data, n_steps):
    """Encode input data into architecture-native MMD features."""
    del n_steps
    return _encode_data_features(encode_component, data)


def check_expansion_mmd(experts, data, threshold, n_steps=8):
    """
    Compare existing experts against the incoming data with the active VAE's
    native latent MMD. The public configuration uses the PSP-based distance.
    """
    del n_steps
    if len(experts) <= 1:
        return True, torch.tensor(float("inf"), device=data.device)

    min_mmd = float("inf")
    num_samples = data.size(0)

    for expert in experts[:-1]:
        z_expert = get_sample_z(
            gen_component=expert,
            encode_component=expert,
            num_samples=num_samples,
            n_steps=expert.n_steps,
        )
        z_data = get_sample_z_from_data(
            encode_component=expert,
            data=data,
            n_steps=expert.n_steps,
        )

        current_mmd = expert.vae.latent_mmd(z_expert, z_data)
        min_mmd = min(min_mmd, current_mmd.item())

    min_mmd_tensor = torch.tensor(min_mmd, device=data.device)
    should_expand = min_mmd > threshold
    return should_expand, min_mmd_tensor


def check_expansion_fire(components, threshold=0.02, num_samples=64, n_steps=8):
    logging.info("=== Torch Expansion Check (threshold=%s, samples=%s) ===", threshold, num_samples)

    if len(components) < 2:
        logging.info("Immediate expansion: component count < 2")
        return True, 0

    new_component = components[-1]
    prev_components = components[:-1]

    mse_values = [
        compute_pairwise_distance(old_comp, new_component, num_samples, n_steps)
        for old_comp in prev_components
    ]
    min_mse = min(mse_values)

    if min_mse > threshold:
        logging.info("Expansion triggered: all existing components are distinct")
        return True, min_mse

    logging.info("No expansion: found similar existing component")
    return False, min_mse
