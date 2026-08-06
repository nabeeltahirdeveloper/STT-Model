"""Roman Urdu auto-captioning.

Pipeline: video -> audio -> fine-tuned Qwen3-ASR -> spelling normalizer ->
forced alignment -> SRT/VTT. Output is Latin script, never Perso-Arabic;
see docs/PROJECT.md §3 for why that is load-bearing rather than cosmetic.
"""

__version__ = "0.1.0"
