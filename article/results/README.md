# Reproducing the article figures

From the repository root, run:

```bash
uv run python examples/experiments/generate_article_figures.py
```

This deterministic run executes only experiments 3, 4, and 5. It writes the
complete plotted metric histories to `article/results`, summary values to CSV,
and PDF/EPS/PNG figures to `article/figures`. Experiment 6 is intentionally not
imported or executed by the generator.

To rebuild the figures from the saved result archives without rerunning the
optimizer, use:

```bash
uv run python examples/experiments/generate_article_figures.py --mode plot
```

The deleted-edge metric is evaluated every 25 outer iterations and once more
at the iteration selected by the minimum implemented objective. The full-flow
metric is stored at every iteration and subsampled every 25 iterations only
for display.
