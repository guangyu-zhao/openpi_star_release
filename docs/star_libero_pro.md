# STAR inference on LIBERO-PRO

This release implements the paper's training-free STAR path for the OpenPI
$\pi_{0.5}$ action expert. Native and STAR inference load exactly the same
training configuration, checkpoint parameters, transforms, and normalization
statistics. STAR only changes hidden-state updates during action sampling.

## Scope and defaults

The intended evaluation scope is the LIBERO-PRO `LIBERO-10-*` family:

- `libero_10_lan`
- `libero_10_object`
- `libero_10_swap`
- `libero_10_task`
- `libero_10_env`

The exact suite keys are provided by the external LIBERO-PRO checkout. The
evaluator deliberately looks them up through LIBERO's benchmark registry rather
than embedding benchmark definitions in this repository.

The release defaults are $\beta_{min}=0.25$, $\beta_{max}=0.55$, and correction
over the final 6 action-expert blocks. The evaluator runs the environment at
10 Hz and executes 5 actions before replanning. It sends every executed
proprioceptive state back to the policy server, so the 2-second STAR statistics
window is based on environment steps rather than only inference calls.

## Installation

Initialize the OpenPI submodules and install the main environment as described
in the top-level README. Install a LIBERO-PRO checkout that registers the five
suite names above in the separate evaluator environment. Checkpoints are not
included in this repository; use either the public `pi05_libero` checkpoint or
a compatible local checkpoint containing its `params/` and `assets/` trees.

## Start the policy server

Native control group with the public checkpoint:

```bash
uv run scripts/serve_policy.py --env LIBERO --inference-mode NATIVE
```

STAR with the same public checkpoint:

```bash
uv run scripts/serve_policy.py \
  --env LIBERO \
  --inference-mode STAR \
  --star-env-control-hz 10 \
  --star-replan-steps 5
```

For a local compatible checkpoint, append the checkpoint subcommand to either
command:

```bash
uv run scripts/serve_policy.py \
  --inference-mode STAR \
  --star-env-control-hz 10 \
  --star-replan-steps 5 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/pi05_libero/checkpoint
```

The beta range and last-six-layer setting are already defaults. They remain
explicit Python/CLI configuration fields so the implementation is inspectable,
but reproducing the release setting does not require passing them.

## Run a LIBERO-PRO suite

From the LIBERO-PRO evaluator environment, with the server listening on port
8000:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_10_lan \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.env-control-hz 10 \
  --args.replan-steps 5
```

Replace `libero_10_lan` with each of the other four suites for the complete
paper scope. Use `--args.task-ids 0,3,7` or `--args.max-tasks 1` for a smoke
test. The client resets policy-side STAR state at every episode boundary and
warns if its control frequency differs from the server metadata.

## Implementation map

- `src/openpi/inference/star/`: phase scheduler and hidden-state momentum core.
- `src/openpi/models/gemma.py`: correction hook in the action-expert blocks.
- `src/openpi/models/pi0.py`: STAR-aware flow-matching sampling.
- `src/openpi/policies/policy.py`: per-step context and episode lifecycle.
- `scripts/serve_policy.py`: native/STAR policy server selection.
- `examples/libero/main.py`: LIBERO-PRO evaluation and executed-state replay.

No STAR-specific training configuration is needed because STAR does not update
weights.
