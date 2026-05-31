# DMSDE

Official implementation for **Dual-Memory Spiking Dynamic Experts (DMSDE)**, a task-free continual learning framework for non-stationary data streams.

DMSDE learns from a stream without using task identities during training or inference. The framework keeps a short-term memory for recent samples and a long-term reference memory for consolidated historical samples. A latent spike-space discrepancy is used to decide whether the current expert should continue adapting or whether a new expert should be initialized. When expansion is not triggered, recent samples are consolidated into long-term memory according to their explanation cost under the active expert.

## Requirements

Python 3.8-3.11 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment.

## Quick Start

Run the default Split MNIST setting:

```bash
python main.py
```

Run Split CIFAR-10:

```bash
python main.py --dataset cifar10
```

## Example

```bash
python main.py \
  --dataset mnist \
  --dataset_fraction 0.15 \
  --n_epochs 20 \
  --batch_size 64 \
  --stream_batch_size 10 \
  --test_batch_size 1 \
  --short_memory_size 128 \
  --long_memory_size 2000 \
  --threshold 0.08 \
  --vae_arch fsvae \
  --classifier_type snn
```

## Scripts

The `scripts/` directory contains convenience launchers:

```bash
bash scripts/run_mnist.sh
bash scripts/run_cifar10.sh
bash scripts/run_all.sh
```

Common environment variables can be overridden:

```bash
GPU=1 DATASET_FRACTION=0.5 CIFAR_N_EPOCHS=50 bash scripts/run_all.sh
```

## Main Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `mnist` | Dataset: `mnist` or `cifar10`. |
| `--dataset_fraction` | `0.15` | Fraction of each split used in experiments. |
| `--num_tasks` | dataset-specific | Number of stream segments. |
| `--vae_arch` | `fsvae` | Spiking variational expert backbone. |
| `--n_epochs` | `20` | Local training epochs for the active expert. |
| `--batch_size` | `64` | Minibatch size for expert training. |
| `--stream_batch_size` | `10` | Batch size of the incoming stream. |
| `--test_batch_size` | `1` | Batch size used by test loaders. |
| `--short_memory_size` | `128` | Short-term memory capacity. |
| `--long_memory_size` | `2000` | Long-term reference memory capacity. |
| `--threshold` | `0.08` | Expansion threshold for the dual-memory discrepancy score. |
| `--n_steps` | `16` | Number of spiking time steps. |
| `--classifier_type` | dataset-specific | Classifier head used by each expert. |
| `--model_dir` | `modelpth` | Directory for checkpoints. |
| `--save_dir` | `results` | Directory for logs and experiment outputs. |

Dataset-specific defaults:

| Dataset | Local epochs | Classifier | Long memory | Threshold |
| --- | --- | --- | --- | --- |
| Split MNIST | `20` | `snn` | `2000` | `0.08` |
| Split CIFAR-10 | `50` | `resnet10` | `1000` | `0.08` |

## Outputs

Training writes logs and results to:

```text
results/YYYY-MM-DD/...
```

Checkpoints are written to:

```text
modelpth/stream_X_model.pth
modelpth/model_final.pth
```

If an unfinished stream checkpoint exists, training resumes from the next stream. If only `model_final.pth` exists, the previous run is treated as complete and a fresh run starts.

## Project Structure

```text
main.py                  Main training entry point
config/config.py         Command-line configuration
data/dataloaders.py      Dataset stream construction
models/component.py      Dynamic expert wrapper
models/classifier.py     Classifier heads
models/vae_fsvae.py      Default spiking variational expert
fsvae_models/            Spiking variational backbone modules
training/trainer.py      Training and checkpoint utilities
utils/memory.py          Dual-memory buffer and expansion signal
utils/testing.py         Evaluation utilities
scripts/                 Dataset launch scripts
```

## License

This code is released for academic research use.
