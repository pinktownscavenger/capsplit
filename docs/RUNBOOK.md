# Runbook: local language-model experiment

How to run the LM experiment (`docs/lm_design.md`, pre-registration `docs/preregistration_lm.md`) from a clean clone. Decisions cited as L-numbers are in `docs/DECISIONS.md`.

## 1. Requirements

- **Machine:** an Apple Silicon Mac is recommended (the MPS GPU backend). Other machines fall back to the CPU, which is much slower.
- **Python 3.10–3.12.** torch 2.8.0 may not install on 3.13 or newer. The commands below use `python3.12` (on macOS: `brew install python@3.12`); `python3.10` or `python3.11` work the same way.
- **Disk:** about 2 GB for the two pinned models in the Hugging Face cache (`~/.cache/huggingface`), plus room for results.
- **Memory:** at least `memory.min_free_gb` free (3 GB in the shipped configs); every stage refuses to start below it.
- **Why torch is pinned to 2.8.0 (L2):** newer torch (2.15.0) aborts on macOS 14 MPS even on a plain matmul (`-[MPSNDArray arrayViewWithShape:strides:]` unrecognized selector). The `lm` extra pins the direct LM dependencies (torch, transformers, peft, huggingface_hub, safetensors, psutil) to the versions verified on the authors' machine. Do not upgrade them. Transitive dependencies (e.g. numpy, tokenizers, accelerate) are not pinned and resolve at install time; every stage manifest records the torch and transformers versions actually used.

## 2. Setup

```bash
git clone https://github.com/sazandkhalid/capsplit.git
cd capsplit
git checkout harness/capability-vs-reliability
git log -1 --format=%H                 # to reproduce the published lm_main results, compare with the
                                       # `commit` in docs/lm_main/measure/manifest.measure.json
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,lm,viz,lint]"   # lint = ruff, for section 8
```

## 3. Running the experiment

Every command has the form `python -m capsplit.harness.lm <stage> <config> [--out results] [--fallback]`. Outputs go to `<out>/<name>/`, where `name` comes from the config (`results/lm_demo/`, `results/lm_main/`). Each stage prints `wrote <path>` for the files it wrote and a `next:` line with the command to run next.

Run the demo first. With the model already downloaded its compute takes a few minutes; `fetch` downloads about 1.6 GB the first time, which can take much longer. It checks the whole pipeline:

```bash
C=configs/lm_demo.json
python -m capsplit.harness.lm doctor    $C   # doctor.json: versions, device, free memory and disk
python -m capsplit.harness.lm fetch     $C   # fetch.json: downloads the pinned model and fallback
python -m capsplit.harness.lm speedtest $C   # speedtest.json: load time, sampling rate, one LoRA step
python -m capsplit.harness.lm data      $C   # data/problems.jsonl, data/summary.json (CPU only)
python -m capsplit.harness.lm base      $C   # samples/base/*.jsonl, base/classes.json, base/summary.json
python -m capsplit.harness.lm train     $C   # train/<arm>/seed<s>/step<k>/adapter.safetensors,
                                             #   timing.json, training.json
python -m capsplit.harness.lm measure   $C   # samples/<arm>/seed<s>/step<k>/eval_T*.jsonl,
                                             #   measure/results.json, null_check.json, budget_curve.json
python -m capsplit.harness.lm report    $C   # report/report.md, summary.json, tables/*.csv, figures/*.png
python -m capsplit.harness.lm verdicts  $C   # report/verdicts.md, verdicts.json
python -m capsplit.harness.lm discovery $C   # report/discovery_lm.md, discovery_lm.json
python -m capsplit.harness.lm sensitivity $C # report/sensitivity_p0.md, sensitivity_p0.json (post-hoc)
```

- `setup` runs `fetch` and then `speedtest` in one command; use either `setup` or the two separate commands.
- `doctor` prints only the file it wrote. Open `doctor.json` and check that the device is `mps` (or `cpu` if you expect it) and that free memory is above `memory.min_free_gb`.
- `base` samples the base model to classify each problem as reachable, unreachable or ambiguous. It also draws the measurement, temperature-sweep and null-check sets.
- `train` trains one LoRA adapter for each arm × seed and saves one at each checkpoint.
- `measure` samples each trained adapter and computes the split, the temperature baseline, the bootstrap intervals, the null check and the budget curve. By default it measures only the final checkpoint. Add `--all-checkpoints` to measure the intermediate ones too.
- `verdicts` reads the thresholds from `configs/lm_prereg.json` (override with `--prereg`). The verdicts are pre-registered only for `configs/lm_main.json`.
- `discovery` compares each RL training problem's first success with the waiting-time model (descriptive; DECISIONS L22).
- `sensitivity` is post-hoc (DECISIONS L21): it diagnoses the null check from the saved samples and re-applies the L1–L3 rules with the base rate taken from set B, from the null set, and from both pooled. It never changes `verdicts.md`.
- `report`, `verdicts`, `discovery` and `sensitivity` only read files; they do not load the model and take seconds.
- `scripts/publish_lm_report.sh results/lm_main docs/lm_main` copies a finished run's generated outputs (reports, summaries, training logs, timings; not samples or adapters) into `docs/lm_main/`, runs the independent check `scripts/rederive_lm_headlines.py` (output `docs/lm_main/rederive_headlines.txt`; set `PYTHON` to the venv's python if `python` is not it), and assembles `docs/report_lm.md` from the generated markdown.
- The stages that load the model or write data (`speedtest`, `data`, `base`, `train`, `measure`) write a `manifest.<stage>.json` next to their outputs (commit, config, seeds, device, versions); each training checkpoint also gets `manifest.step.json`. Every stage appends its wall time to `timing/<stage>.jsonl`. The read-only stages (`report`, `verdicts`, `discovery`, `sensitivity`) write no manifest.
- **Expected messages that are harmless:** torch's `Redirects are currently not supported in Windows or MacOs` on every stage, a `requires_grad` UserWarning from `speedtest`, and Hugging Face download progress bars.
- **Expected warnings on the demo:** with only 10 problems the demo usually has an empty class (often `ambiguous`), and `base` and the report print a large warning about it; L1 is then typically inconclusive for having too few problems. That is expected for the demo and does not mean the run is broken. `verdicts.md` is titled "pre-registered" for every config, but the thresholds are pre-registered only for `configs/lm_main.json`; the file says so.

Then run the full experiment with `C=configs/lm_main.json` and the same commands. It takes several hours on an M3 Pro. The run records its own durations: training wall time is in `train/<arm>/seed<s>/timing.json`, and progress lines with elapsed time and ETA go to stderr, so keep a log (section 4). `lm_main.json` is frozen with the pre-registration (tag `lm-prereg-frozen`). Do not edit it. Make a copy with a new `name` instead.

## 4. Keeping long runs alive

- Wrap long stages in `caffeinate -i` (L13). This stops idle sleep while the command runs. Plug in power and **keep the lid open**: closing the lid sleeps the machine whatever caffeinate does.
- `caffeinate -i` alone did not stop macOS "maintenance sleep" in one test run: the machine was idle with its display off when the stage started, and it slept for about 15 minutes mid-stage (the stage then resumed by itself; nothing was lost). To prevent this, also run `caffeinate -dimsu` in a separate terminal for the duration of the run (stop it with Ctrl-C afterwards), and start runs with the machine awake.
- To run in the background with a log:

  ```bash
  nohup caffeinate -i python -m capsplit.harness.lm train configs/lm_main.json > train.log 2>&1 &
  tail -f train.log
  ```

- You do not need to add `nice`. Each stage applies the config's `runtime` section itself (`nice`, added niceness, and `threads`, torch CPU threads), so the machine stays usable.

## 5. Stopping and resuming

You can stop any stage at any time (Ctrl-C, `kill`, sleep, crash). To resume, re-run **the same command**.

- **data** is regenerated deterministically, and an existing file is checked against the config.
- **base / measure** resume per sample: only the missing (problem, sample index) rows are generated, and finished rows are kept. The statistics are recomputed on each run, which is cheap.
- **train**: an arm × seed is finished once its final adapter and `training.json` exist, and finished runs are skipped. An interrupted run **restarts from step 0**, because optimizer state is not saved (L15).
- **Stale-output guard (L19):** each manifest stores a hash for each config section. If you change a section, or switch `--fallback` on or off, a stage that would reuse older outputs exits with code 2 and names the sections that differ, for example `differs ... in: sampling`. Either give the config a new `name` or delete the path the message names. Give `--fallback` to every stage or to none.

## 6. Shrinking the run for a slower machine

Copy the config, give it a **new `name`** (smaller sizes are a different experiment, and the pre-registered verdicts no longer apply), then lower these keys, in this order:

1. `seeds`, for example `[0]`.
2. Drop the optional `sft_then_rl` arm from `arms`.
3. Samples per problem: `sampling.n_classify`, `n_measure`, `n_trained`, `n_temperature`. A smaller `n_classify` lowers the largest N the budget curve can resolve.
4. Number of problems: `task.generators[*].n_train` / `n_heldout`.

If you run out of memory, lower these:

- `sampling.batch_size`: sequences per forward pass.
- the `lm_rl` param `micro_batch` (and `rl.micro_batch` in `sft_then_rl`).
- `group` × `problems_per_update`: completions per RL update.
- `model.max_new_tokens`.
- Or pass `--fallback` to every stage to use the smaller model (SmolLM2-360M).

## 7. Errors (exit code 2)

Problems you can fix print `error: ...` and exit with code 2. Finished work is always kept.

| Message says | Do this |
|---|---|
| `only X GB of memory is available, below memory.min_free_gb` | Close other apps and re-run. Or use the memory knobs above or `--fallback`. |
| `this process uses X GB, above memory.max_process_gb` | Lower `sampling.batch_size`, `micro_batch` or the group size, then re-run. |
| `ran out of MPS` / `ran out of memory` | Lower, in the order the message lists: `micro_batch`, `sampling.batch_size`, group × problems_per_update, the SFT `batch`, `max_new_tokens`. Raise `memory.max_process_gb` only if the memory is really free. |
| `... does not exist; run the <stage> stage first` / `has not finished training` | Run the stage it names. |
| `produced with a config that differs ... in: <sections>` / `exists and differs` | Use a new `name`, or delete the path it names (section 5). |
| `model.device is 'mps' but PyTorch reports MPS unavailable` | Set `model.device` to `"auto"` or `"cpu"`. |
| `OSError ... run the fetch stage first` | Run `fetch`. If that fails, check the network. |
| Config errors (unknown key, bad value) | Fix the config. Unknown keys are rejected on purpose. |

## 8. Tests and lint

```bash
pytest -m "not lm_model"                     # everything except the real-model tests (about 8 minutes)
pytest -m lm_model                           # needs the weights downloaded by `fetch`; skips otherwise
ruff check src/capsplit/harness/lm tests/lm  # lint gate for the LM code (L4); needs the lint extra
```

## 9. The toy experiment

The original toy experiment has its own CLI, `python -m capsplit.harness` (`list`, `run`, `report`, `verdicts`). `docs/harness.md` documents it. That experiment is frozen: do not change its code or configs.
