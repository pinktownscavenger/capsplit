# Step 0 audit (read-only; nothing was changed)

These are findings about the completed `lm_main` run and its code. They are reported, not fixed.

## (a) Precision of the trainable LoRA parameters and of the optimizer steps

**What was checked:**
- **The loading path.** `lm/model.py: load` loads the base in `model.dtype` = float16 on MPS (`lm/device.py: dtype_of`; CPU always uses float32). `attach_lora` then calls `peft.get_peft_model` with peft's default `autocast_adapter_dtype=True`.
- **A load test.** The model was loaded in float16 and `attach_lora` applied. The base parameters are float16 and frozen; every LoRA parameter is float32 and trainable.
- **The saved adapters.** Every saved `adapter.safetensors` in `results/lm_main/train/` stores float32 tensors.

**What follows:**
- **Optimizer steps are applied in float32.** `lm/trainers.py: _optimizer` builds `torch.optim.AdamW` over the LoRA parameters only, so its state and updates are float32. Gradients are clipped to norm 1.0 before each step.
- **No mixed-precision machinery is used:** no `torch.autocast` and no `GradScaler` anywhere in `lm/` or `trainers/`.
- **The forward pass is mixed.** peft's LoRA `Linear.forward` (peft 0.22.0, `tuners/lora/layer.py`) does three things in order:
  - casts the layer input to float32;
  - computes B·A·x in float32 and adds it to the float16 base output;
  - casts the sum back to float16.

  So the adapter's contribution passes through float16 rounding in every layer's output. Gradients reach the adapter through float16 activations without loss scaling, so very small gradient components could underflow. This was not measured.

## (b) `sampling.top_p`

`configs/lm_main.json` sets no `top_p`, so the config default **1.0** applies (`lm/config.py`). Sampling is untruncated, and training samples were on-policy for the scored distribution. The KL layer samples with the same value.

## (c) What `sft_loss` normalizes by

`trainers/sft.py: sft_loss` returns −mean over the batch of `policy.log_prob(prompt, demonstration)`. `LocalLMPolicy.log_prob` **sums** token log-probs over the demonstration, which is the answer plus the stop token.

So the loss is normalized **per sequence**: it is the mean over sequences of the summed token log-likelihood. It is **not** normalized per token, so longer answers carry more weight per example.

The RL loss (`lm/trainers.py: accumulate_pg_gradient`) is likewise −Σ_i A_i · ln π(y_i) / n, per sequence with summed token log-probs.

## (d) Adapter files on disk

The config's checkpoints are 0, 25, 50, 100, 200, for arms `none`, `rl`, `sft`, `sft_then_rl` and seeds 0, 1, 2. **Every one** of these `results/lm_main/train/<arm>/seed<s>/step<k>/adapter.safetensors` files exists, so nothing is missing and nothing was retrained.

The KL stage records every adapter it reads, with its SHA-256, in `manifest.kl.json`. The KL report's `tables/weights_adapters.csv` has one row per adapter found on disk.

## Gate 0

Trained adapters load. Every final-checkpoint adapter of `rl`, `sft` and `sft_then_rl` (seeds 0, 1, 2) was loaded into the real model with `LocalLMPolicy.load_adapter`, and the timing pilot loaded and sampled the `none` adapters. **Gate 0 passed.**
