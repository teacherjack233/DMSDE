import logging
import os
import re

import torch
import torch.nn as nn

from models.component import Component


TRAINING_COMPLETE_MARKER = "TRAINING_COMPLETE"
TRAINING_IN_PROGRESS_MARKER = "TRAINING_IN_PROGRESS"


def train_component(component, memory_samples, memory_labels, batch_size, n_epochs=20):
    """Train the active expert on the current DMSDE memory view."""
    total_samples = memory_samples.size(0)
    if total_samples == 0:
        return

    num_batches = (total_samples + batch_size - 1) // batch_size
    logging.info("Training expert on %s samples (%s batches)", total_samples, num_batches)

    component.train()
    for epoch in range(n_epochs):
        component.vae.update_p(epoch, n_epochs)
        last_vae_loss = 0.0
        last_classifier_loss = 0.0

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_data = memory_samples[start_idx:end_idx]
            batch_labels = memory_labels[start_idx:end_idx]

            last_vae_loss = component.train_vae(batch_data)
            last_classifier_loss = component.train_classifier(batch_data, batch_labels)

        if (epoch + 1) % max(1, min(9, n_epochs)) == 0:
            message = (
                f"Epoch [{epoch + 1}/{n_epochs}] "
                f"VAE Loss: {last_vae_loss:.4f} "
                f"Classifier Loss: {last_classifier_loss:.4f}"
            )
            print(message)
            logging.info(message)


def handle_memory_overflow(
    memory,
    components,
    component,
    flat_data,
    labels,
    device,
    time_dir,
    strategy="diversity",
    input_channels=1,
):
    """Legacy single-buffer overflow handler kept for old experiment scripts."""
    if strategy == "sliding_window" or len(components) == 0:
        num_new_samples = len(flat_data)
        new_entries = list(zip(
            flat_data.detach().cpu().tolist(),
            labels.detach().cpu().tolist(),
        ))
        memory.buffer = memory.buffer[num_new_samples:] + new_entries
        return component

    current_samples, current_labels = memory.get_samples()
    if current_samples.size(1) != input_channels:
        if input_channels == 3 and current_samples.size(1) == 1:
            current_samples = current_samples.repeat(1, 3, 1, 1)
        elif input_channels == 1 and current_samples.size(1) == 3:
            current_samples = current_samples.mean(dim=1, keepdim=True)

    combined_samples = torch.cat([current_samples.cpu(), flat_data.detach().cpu()], dim=0).to(device)
    combined_labels = torch.cat([current_labels.cpu(), labels.detach().cpu()], dim=0).to(device)
    num_samples = len(combined_samples)

    classifiers = [comp.classifier for comp in components]
    with torch.no_grad():
        scores = torch.zeros(num_samples, device=device)
        for classifier in classifiers:
            classifier.eval()
            outputs = classifier(combined_samples)
            scores += nn.CrossEntropyLoss(reduction="none")(outputs, combined_labels)
        scores /= max(len(classifiers), 1)

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    _, sorted_indices = torch.sort(scores, descending=True)
    selected_indices = sorted_indices[:memory.size]
    selected_samples = combined_samples[selected_indices].cpu()
    selected_labels = combined_labels[selected_indices].cpu()
    memory.buffer = list(zip(selected_samples.tolist(), selected_labels.tolist()))
    return component


def create_new_component(args, components):
    """Create and register a fresh DMSDE expert."""
    num_classes = 10
    component = Component(
        gan_z_dim=128,
        learning_rate=0.0001,
        beta1=0.5,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        input_channels=args.input_channels,
        num_classes=num_classes,
        img_size=args.img_size,
        classifier_type=args.classifier_type,
        vae_arch=getattr(args, "vae_arch", "fsvae"),
        vae_mmd_type=getattr(args, "vae_mmd_type", "rbf"),
        vae_distance_lambda=getattr(args, "vae_distance_lambda", 0.001),
        feature_mode=False,
    )
    components.append(component)
    message = f"Created expert #{len(components)} ({component.vae_arch.upper()})"
    print(message)
    logging.info(message)
    return component


def _num_classes_for_args(args):
    del args
    return 10


def _build_component_for_load(
    args,
    classifier_type=None,
    vae_arch=None,
    vae_mmd_type=None,
    vae_distance_lambda=None,
    feature_mode=None,
):
    return Component(
        gan_z_dim=128,
        learning_rate=0.0001,
        beta1=0.5,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        input_channels=args.input_channels,
        num_classes=_num_classes_for_args(args),
        img_size=args.img_size,
        classifier_type=classifier_type or getattr(args, "classifier_type", "ann"),
        vae_arch=vae_arch or getattr(args, "vae_arch", "fsvae"),
        vae_mmd_type=vae_mmd_type or getattr(args, "vae_mmd_type", "rbf"),
        vae_distance_lambda=(
            vae_distance_lambda
            if vae_distance_lambda is not None
            else getattr(args, "vae_distance_lambda", 0.001)
        ),
        feature_mode=(
            feature_mode
            if feature_mode is not None
            else False
        ),
    )


def find_latest_checkpoint(model_dir, num_tasks):
    os.makedirs(model_dir, exist_ok=True)
    in_progress_path = os.path.join(model_dir, TRAINING_IN_PROGRESS_MARKER)
    marker_path = os.path.join(model_dir, TRAINING_COMPLETE_MARKER)
    final_model_path = os.path.join(model_dir, "model_final.pth")
    in_progress_mtime = (
        os.path.getmtime(in_progress_path)
        if os.path.exists(in_progress_path)
        else None
    )
    if not os.path.exists(in_progress_path) and (
        os.path.exists(marker_path) or os.path.exists(final_model_path)
    ):
        logging.info(
            "Completed training output found in %s; skipping automatic checkpoint resume",
            model_dir,
        )
        return None, 0

    pth_files = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".pth")
    ]
    if in_progress_mtime is not None:
        pth_files = [
            path for path in pth_files
            if os.path.getmtime(path) >= in_progress_mtime
        ]
    if not pth_files:
        return None, 0

    stream_checkpoints = []
    for path in pth_files:
        match = re.search(r"stream_(\d+)_model\.pth$", os.path.basename(path))
        if match:
            stream_checkpoints.append((int(match.group(1)), path))

    if stream_checkpoints:
        stream_id, path = max(stream_checkpoints, key=lambda item: item[0])
        return path, stream_id

    return None, 0


def mark_training_started(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    in_progress_path = os.path.join(model_dir, TRAINING_IN_PROGRESS_MARKER)
    with open(in_progress_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("in_progress\n")
    logging.info("Wrote training-in-progress marker to %s", in_progress_path)


def mark_training_complete(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    in_progress_path = os.path.join(model_dir, TRAINING_IN_PROGRESS_MARKER)
    marker_path = os.path.join(model_dir, TRAINING_COMPLETE_MARKER)
    if os.path.exists(in_progress_path):
        os.remove(in_progress_path)
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("complete\n")
    logging.info("Wrote training-complete marker to %s", marker_path)


def clear_training_complete_marker(model_dir):
    marker_path = os.path.join(model_dir, TRAINING_COMPLETE_MARKER)
    if os.path.exists(marker_path):
        os.remove(marker_path)
        logging.info("Removed stale training-complete marker from %s", marker_path)


def load_model_state(checkpoint_path, args):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    component_items = []
    for key, value in checkpoint.items():
        match = re.match(r"component_(\d+)$", key)
        if match:
            component_items.append((int(match.group(1)), value))

    components = []
    for _, state in sorted(component_items, key=lambda item: item[0]):
        saved_classifier_type = state.get("classifier_type")
        saved_vae_arch = state.get("vae_arch")
        saved_vae_mmd_type = state.get("vae_mmd_type", getattr(args, "vae_mmd_type", "rbf"))
        saved_vae_distance_lambda = state.get(
            "vae_distance_lambda",
            getattr(args, "vae_distance_lambda", 0.001),
        )
        saved_feature_mode = state.get(
            "feature_mode",
            False,
        )
        vae_arch_candidates = [
            saved_vae_arch,
            getattr(args, "vae_arch", None),
            "fsvae",
        ]
        vae_arch_candidates = [
            item for index, item in enumerate(vae_arch_candidates)
            if item is not None and item not in vae_arch_candidates[:index]
        ]
        classifier_candidates = [
            saved_classifier_type,
            getattr(args, "classifier_type", None),
            "ann",
            "snn",
            "resnet18",
        ]
        classifier_candidates = [
            item for index, item in enumerate(classifier_candidates)
            if item is not None and item not in classifier_candidates[:index]
        ]

        component = None
        last_error = None
        for vae_arch in vae_arch_candidates:
            for classifier_type in classifier_candidates:
                candidate = _build_component_for_load(
                    args,
                    classifier_type=classifier_type,
                    vae_arch=vae_arch,
                    vae_mmd_type=saved_vae_mmd_type,
                    vae_distance_lambda=saved_vae_distance_lambda,
                    feature_mode=saved_feature_mode,
                )
                try:
                    candidate.vae.load_state_dict(state["vae"])
                    candidate.classifier.load_state_dict(state["classifier"])
                except RuntimeError as error:
                    last_error = error
                    continue
                component = candidate
                break
            if component is not None:
                break

        if component is None:
            raise RuntimeError(
                f"Failed to load component from {checkpoint_path}. "
                f"Last error: {last_error}"
            )

        if state.get("frozen", False):
            component.freeze()
        components.append(component)

    active_component = None
    for component in reversed(components):
        if not component.is_frozen:
            active_component = component
            break
    if active_component is None and components:
        active_component = components[-1]

    logging.info("Loaded %s experts from %s", len(components), checkpoint_path)
    print(f"Loaded {len(components)} experts from {checkpoint_path}")
    return components, active_component


def freeze_component(component):
    component.freeze()
    logging.info("Frozen current expert")


def save_model_state(components, model_path, model_filename="model.pth"):
    os.makedirs(model_path, exist_ok=True)
    model_filepath = os.path.join(model_path, model_filename)
    torch.save({
        f"component_{i + 1}": {
            "vae": comp.vae.state_dict(),
            "classifier": comp.classifier.state_dict(),
            "vae_arch": comp.vae_arch,
            "vae_mmd_type": comp.vae_mmd_type,
            "vae_distance_lambda": comp.vae_distance_lambda,
            "feature_mode": comp.feature_mode,
            "classifier_type": comp.classifier_type,
            "frozen": comp.is_frozen,
        }
        for i, comp in enumerate(components)
    }, model_filepath)
    logging.info("Saved model to %s", model_filepath)
