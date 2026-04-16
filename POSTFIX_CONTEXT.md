# Post-Bugfix Context for Remote Session

## The Bug (ALREADY FIXED in this codebase)

In `minicolossal/plugin.py`, the `configure()` method had an `if/elif` that made
Tensor Parallelism (TP) dead code whenever Pipeline Parallelism (PP) was active:

```python
# BEFORE (broken):
if self.pp_size > 1:
    model = create_pipeline_stage(...)      # uses plain TransformerBlock
elif self.tp_size > 1:
    model = TensorParallelGPT2(...)         # NEVER reached when pp > 1
```

When running `dp=2, pp=2, tp=2` on 8 GPUs, only PP was active. The TP process
groups were created in the mesh but carried zero traffic. Effectively, the system
ran two independent `DP(2)×PP(2)` training runs, with 4 GPUs duplicating work
instead of doing unique TP-sharded computation.

## The Fix (ALREADY APPLIED to files on this machine)

Three files were modified. The code has already been synced to this master node
AND the launch scripts will rsync to the worker automatically.

### 1. `minicolossal/pipeline_parallel.py`
- Added `TensorParallelPipelineStage` class (line ~163): same as `PipelineStage`
  but uses `TensorParallelTransformerBlock` (TP-sharded attention + MLP)
- Added `create_tp_pipeline_stage()` factory function

### 2. `minicolossal/t5.py`
- Added `TensorParallelT5PipelineStage` class (line ~398): same as
  `T5PipelineStage` but uses `TensorParallelT5EncoderBlock` /
  `TensorParallelT5DecoderBlock`
- Added `create_tp_t5_pipeline_stage()` factory function

### 3. `minicolossal/plugin.py`
- Fixed `configure()` (line 276): new `if self.pp_size > 1 and self.tp_size > 1`
  branch that creates TP-sharded pipeline stages
- Added TP sync (lines 309-326): broadcasts replicated parameters (embeddings,
  LayerNorm, lm_head) from `tp_rank=0` within each TP group. TP-sharded params
  (ColumnParallelLinear, RowParallelLinear) are intentionally excluded.
- Updated imports: `create_tp_pipeline_stage`, `create_tp_t5_pipeline_stage`
- Updated docstring to list 3D configs

## What Needs to Be Done

Run the post-bugfix experiments and collect results. The experiment script is
already created at `benchmarks/run_phase3_postfix.sh`.

### Setup
```bash
export MASTER_ADDR=10.0.3.199
export WORKER_IP=10.0.3.146
cd /home/ubuntu/workspace/mini-colossal-ai
```

### Run all experiments
```bash
bash benchmarks/run_phase3_postfix.sh
```

This runs 8 experiments (results saved to `results/phase3_postfix/`):

| ID | Config | Model | Batch | Placement |
|----|--------|-------|-------|-----------|
| B1 | dp2,pp2,tp2 | GPT-2 Medium | 8 | Good (PP inter-node) |
| B2 | dp2,pp2,tp2 | GPT-2 Medium | 16 | Good (PP inter-node) |
| C1 | dp2,pp2,tp2 | GPT-2 Medium | 8 | Bad (DP inter-node) |
| D1 | dp2,pp2,tp2 | GPT-2 Medium | 8 | Worst (TP inter-node) |
| E1 | dp2,pp2,tp2 | GPT-2 Small | 8 | Good |
| E2 | dp2,pp2,tp2 | GPT-2 Medium | 8 | Good |
| E3 | dp2,pp2,tp2 | T5-base | 8 | Good |
| F1 | dp2,pp2,tp2+ZeRO1 | GPT-2 Medium | 8 | Good |

### After experiments complete

1. Fill in results in `results/phase3_postfix/POSTFIX_RESULTS_SUMMARY.txt`
2. Update data in `generate_figures_postfix.py` with actual numbers
3. Run `python generate_figures_postfix.py` to create comparison figures
4. Sync results back: `rsync results/ and figures/ back to local`

## Pre-Fix Baseline Numbers (for comparison)

These are the results from the BROKEN code (TP was inactive):

| Config | Model | Throughput | Speedup vs 1-GPU |
|--------|-------|-----------|-------------------|
| Single GPU | GPT-2 Small | 3,879 tok/s | 1.0× |
| Single GPU | GPT-2 Medium | 1,404 tok/s | 1.0× |
| Single GPU | T5-base | 3,359 tok/s | 1.0× |
| 3D Good (broken) | GPT-2 Small | 8,678 tok/s | 2.24× |
| 3D Good (broken) | GPT-2 Medium | 3,519 tok/s | 2.51× |
| 3D Good (broken) | T5-base | 5,790 tok/s | 1.72× |
| 3D Bad (broken) | GPT-2 Medium | 2,926 tok/s | 2.08× |
| 3D TP-inter (broken) | GPT-2 Medium | 3,543 tok/s | 2.52× |
| 3D+ZeRO1 (broken) | GPT-2 Medium | 3,044 tok/s | 2.17× |

## Expected Behavior After Fix

- **Throughput may be slightly lower** at the same batch size because TP now adds
  real all-reduce communication (2 per block for GPT-2, 3 per block for T5 decoder)
  that wasn't happening before.
- **Memory per GPU should drop** because TP shards the attention and MLP weights
  across 2 ranks. This enables larger batch sizes (B2 with bs=16).
- **Training is now CORRECT** — all 8 GPUs contribute unique work.
- **Max theoretical speedup is ~4×** (DP×PP = 2×2). TP does not multiply throughput;
  it only reduces memory.
- **Placement impact should be MORE pronounced** with real TP traffic, especially
  for "worst placement" (TP inter-node) which now has real all-reduces over TCP.

## Troubleshooting

If an experiment hangs or crashes:
- Check worker connectivity: `ssh -i /home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem ubuntu@10.0.3.146 hostname`
- Kill stale processes: `pkill -9 -f bench_unified` on both nodes
- If OOM on bs=16, try bs=12
- For T5 (E3), if it fails, it may be because T5 PP only supports pp_size=2
  (encoder=stage0, decoder=stage1), which is what we're using — should work
