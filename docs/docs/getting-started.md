# Getting Started

## Requirements

- Python >= 3.8
- CUDA GPU with at least **8 GB VRAM** for inference-only techniques; **16 GB+** for training-based techniques (ESD, CoGFD, AdvUnlearn)

!!! info "VRAM requirements"
    VRAM needs vary significantly across techniques — inference-only methods need ~5 GB
    while training-based methods with frozen model copies can peak at 12–16 GB during
    training. See [GPU Requirements](running-experiments/gpu-requirements.md) for a full
    per-technique breakdown.

## Installation

### 1. Install eval-unlearn

```bash
pip install eval-unlearn
```

### 2. Install technique packages

To ensure of a lightweight and clean package, the precise implementation of all unlearning techniques are in  seperate installable packages hosted on [Hugging Face](https://huggingface.co/datasets/Unlearningltd/Packages).

Some packages bundle large weight files tracked via Git LFS. Installing via `pip install git+...` will give you a broken pointer file instead of the real weights. Clone the packages repo once and install editably:

```bash
git clone https://huggingface.co/datasets/Unlearningltd/Packages
cd Packages
git lfs pull
```

Then install only the ones you need:

```bash
pip install -e esd/
pip install -e mace/
pip install -e uce/
pip install -e ssd/
pip install -e ca/
pip install -e cogfd/
pip install -e trasce/
pip install -e saeuron/
pip install -e safree/
pip install -e concept-steerers/
pip install -e advunlearn/
```

SLD is included in `eval-unlearn` directly and requires no extra install as it is implemented within the [Hugging-Face] [diffusers] library, a required dependency of the package.

### 3. Metric packages.

Most metrics are fairly lightweight and their implementation does not require any standalone dependencies. ASR I2P works out of the box for all supported I2P concepts. For adversarial evaluation, `asr_ring_a_bell` and `asr_mma_diffusion` use separate prompt generation techniques to discover adversarial prompts — these require additional packages to be installed.

From the cloned `Packages` directory (see step 2 above):

```bash
pip install -e p4d/
pip install -e mma_diff/
pip install -e RING_A_BELL/
pip install -e Q16/
```

### 4. Metric extras

Some metrics require additional dependencies:

```bash
pip install "eval-unlearn[asr]"    # ASR — requires NudeNet
pip install "eval-unlearn[fid]"    # FID — requires torchvision
pip install "eval-unlearn[coco]"   # COCO-based metrics — requires torchvision
```

### CUDA wheels

If installing PyTorch via pip rather than conda:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu126
```

## Hugging Face authentication

Most techniques download model weights from the Hugging Face Hub.
Create a `.env` file at the project root:

```bash
HF_TOKEN=your_token_here
```

Eval-Unlearn loads this automatically on startup. Alternatively export it in your shell.

## Running a benchmark

Benchmarks are defined in a config file and run with:

```bash
eval-unlearn run --config config.yaml   # or config.json
```

!!! warning "Check compatibility before running"
    Not all technique–metric pairs are valid. Before writing your config, see [Compatibility](running-experiments/compatibility.md) to confirm your combination is supported.

Both YAML and JSON are supported and equivalent.

### Single metric

=== "YAML"

    ```yaml
    technique:
      name: mace
      config:
        erase_concept: nudity
        lambda_cfr: 0.1
        save_path: checkpoints/mace_nudity.pt

    metric:
      name: asr

    output_dir: results/mace_nudity
    ```

=== "JSON"

    ```json
    {
      "technique": {
        "name": "mace",
        "config": {
          "erase_concept": "nudity",
          "lambda_cfr": 0.1,
          "save_path": "checkpoints/mace_nudity.pt"
        }
      },
      "metric": {
        "name": "asr_i2p"
      },
      "output_dir": "results/mace_nudity"
    }
    ```

### Multiple metrics

Replace `metric` with `metrics` as a list:

=== "YAML"

    ```yaml
    technique:
      name: mace
      config:
        erase_concept: nudity

    metrics:
      - name: asr
      - name: clip_score
      - name: fid

    output_dir: results/mace_nudity
    ```

=== "JSON"

    ```json
    {
      "technique": {
        "name": "mace",
        "config": {
          "erase_concept": "nudity"
        }
      },
      "metrics": [
        { "name": "asr_i2p" },
        { "name": "clip_score" },
        { "name": "fid" }
      ],
      "output_dir": "results/mace_nudity"
    }
    ```

Results are written to `output_dir` as JSON.

## Pushing results to HF Hub

Run and push in one step:

```bash
eval-unlearn run --config config.yaml --hf-repo your-org/results
```

Or push an existing results directory separately:

```bash
eval-unlearn push --repo your-org/results --local-dir results/mace_nudity
```

## Useful commands

List all installed techniques and metrics:

```bash
eval-unlearn plugins
```

Show the base model each technique uses:

```bash
eval-unlearn models
```

## Next steps

- [Concepts](concepts.md) — what unlearning benchmarking measures
- [Techniques](techniques/index.md) — configuration reference for each technique
- [Metrics](metrics/index.md) — what each metric measures and when to use it
