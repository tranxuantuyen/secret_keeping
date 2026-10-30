# Secret Keeping

## Install

Python 3.10+, CUDA PyTorch, and a GPU are required for the supplied 9B-model
experiments.

```bash
conda activate eval_aware
cd /weka/ttuyen/Project/secret_keeping
python -m pip install -e ".[jacobian]"
```

## Basic usage

Run one experiment:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml
```

Override configuration values with `--opts KEY VALUE` pairs:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml --opts \
  DATA.LIMIT 10 METHOD.LAYER 23 INFERENCE.NUM_SAMPLES 2
```


See [INSTRUCTION.md](INSTRUCTION.md) for changing models or adapters, selecting
prompt sets, limiting data, running all organisms, adding a lens, and inspecting
outputs.

## Results

Single-layer (layer 23) accuracy on standard prompts.

| Model organism | Logit Lens | JLens | Template Lens |
|---|---:|---:|---:|
| Gender/female | 99.8% | 99.9% | 91.0% |
| Gender/male | 37.3% | 38.2% | 99.0% |
| Taboo/gold | 84.1% | 82.6% | 91.0% |

### JLens ablation

Accuracy with only the generated response or only the JLens token ranking
provided to the auditor (`n=1000`).

| Model organism | Response only | JLens ranking only |
|---|---:|---:|
| Gender/female | 8.6% | 100.0% |
| Gender/male | 94.8% | 58.3% |
| Taboo/gold | 46.5% | 77.6% |

## TODO

- Test other lens
