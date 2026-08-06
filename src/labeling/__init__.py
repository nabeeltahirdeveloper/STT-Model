"""Label generation: romanization, then canonical spelling normalization.

This is the highest-leverage stage in the project (PROJECT.md §5.3). Model
quality is capped by label quality, and in an end-to-end model, label spelling
inconsistency is baked into the weights permanently -- it cannot be fixed at
inference time. Everything here must be pure and deterministic.
"""
