# FineWeb memory recovery: GPU engineering evidence

Both architecture probes passed on real RTX 4090 allocations on 2026-09-06.
This is **not** a completed 1B experiment or a quality/performance improvement.

| Architecture | Parameters | Peak allocated | Peak reserved | Optimizer updates | Scheduler |
| --- | ---: | ---: | ---: | ---: | --- |
| Deep-thin | 134,515,008 | 8.108 GiB | 8.715 GiB | 3 | `1560874_0`, completed, exit `0:0` |
| Wide | 141,576,960 | 6.929 GiB | 7.486 GiB | 3 | `1560874_1`, completed, exit `0:0` |

The historical deep-thin run OOMed at the first backward pass with microbatch
6. Both new probes use microbatch 2 and accumulation 39, preserving 78 blocks
and 159,744 tokens per optimizer update. Every step has finite loss and a
finite, nonzero gradient norm; sampled weights changed, and one validation
batch produced a finite loss. No checkpoint was written by these probes.
The two allocated devices report different total visible memory (about 22.03
and 23.52 GiB); this is not a matched-hardware speed comparison.

## Provenance

- Execution commit: `8b02955ba11aab21b320a1cec00cdaba1685db18`.
- Deployment archive SHA256: `afd5f18e9086a465a7d72e8a843769710657d52c91f46de9858b3a546b15a029`.
- Source/config hashes were checked on each worker before execution.
- Packed data was fully hash-verified: 1,000,000,750 train tokens and 4,194,596
  validation tokens. Existing data/tokenizer assets were reused read-only.
- Runtime: PyTorch `2.11.0+cu128`. Unchanged training helpers were loaded from
  the separate recovery snapshot, not patched into the original run directory.
- Deep array task `_0` has raw job ID `1560876`, explaining the JSON filename;
  wide array task `_1` has raw job ID `1560874`.
- Original machine-generated JSON and logs are in [probe_20260906](probe_20260906/).

Run `python scripts/verify_fineweb_recovery_probe.py results/fineweb_recovery/probe_20260906`.
The verifier checks pinned payload hashes, frozen execution files, configuration
identity, model size, token budget, optimizer steps and memory accounting.
Regression tests additionally check the exact block sequence per optimizer step
under re-batching and FP64 toy AdamW update parity. Neither test establishes
bitwise BF16 equivalence on GPU.

## Next experiment and current gate

The [frozen protocol](../../protocols/fineweb_memory_matched_recovery.md) defines
two architectures × two schedules × three seeds, each with 999,997,440 tokens.
It covers all 2,048 complete validation blocks, unlike the historical 1,536-block
subset. Old and new endpoints must not be pooled as replicates.

**Full training has not been submitted.** At the post-probe inspection, the
shared 10-TiB `/ssd` volume was 99% used, with 139,236,065,280 bytes free (about
129.67 GiB). The ordinary quota command reported only the unrelated home
filesystem; account-scoped SSD quota/headroom is not verified. Other training
is writing to this shared volume. Safe storage availability must be established
before launching the 12-cell matrix, initially at most four concurrent runs.
No historical checkpoint, dataset, or unrelated job was deleted or changed.
