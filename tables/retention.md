# Retention, looped 4x8 at H=8, mean over hops 1-7 (iterations)

| Probe set | Retention | Paper | Per seed, hops 1-7 (levels) |
|---|---|---|---|
| lr 3e-4, seeds 42-44, prompt-disjoint, per layer / 4 | 2.17 | 2.2 | [[24, 12, 9, 6, 8, 3, 0], [21, 6, 7, 7, 7, 2, 0], [10, 16, 12, 6, 8, 10, 8]] |
| lr 1e-3, seed 42 re-probe, prompt-disjoint, per layer / 4 | 2.61 | 2.6 | [[20, 9, 14, 11, 9, 6, 4]] |
| lr 1e-3, seeds 42-45, row split, per iteration | 1.68 | 1.7 | [[5, 3, 2, 2, 3, 1, 1], [5, 2, 1, 1, 1, 1, 0], [4, 2, 1, 1, 1, 0, 0], [5, 2, 1, 1, 1, 0, 0]] |
