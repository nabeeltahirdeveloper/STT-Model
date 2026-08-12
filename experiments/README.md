# experiments/

One-off investigations. **Not production code, not part of the pipeline.**

`scripts/` holds the CLIs the pipeline actually runs — they are linted, typed,
tested and documented in `docs/RUNBOOK.md`. Code here answers a specific
question once, is allowed to be rougher, and may stop working when the thing it
probes changes. Nothing in `src/` or `scripts/` may import from here.

Each experiment gets its own directory with a README that states:

- the question it was run to answer
- what it concluded
- the ADR that records the conclusion, if there is one

Keep the conclusion here even when the answer was "this does not work". A failed
experiment that is deleted gets repeated; one that is written down does not.
See ADR-014, where a promising corpus-alignment idea was wrong in a way that
took a while to see.

| Directory | Question |
|---|---|
| `lora_mps/` | Does the label pipeline teach Roman output? (ADR-016) |
