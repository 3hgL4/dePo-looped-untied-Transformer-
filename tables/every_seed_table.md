# Table A1: first decodable layer (H=4) and full-answer first-query accuracy, every seed

| Model, lr | Probes | Seed | Hop 1 | Hop 2 | Hop 3 | Answer | Accuracy |
|---|---|---|---|---|---|---|---|
| Looped 4x8, 1e-3 | row split | 42 | 8 | 12 | 16 | 16 | 1.000 |
| Looped 4x8, 1e-3 | row split | 43 | 8 | 12 | 16 | 20 | 1.000 |
| Looped 4x8, 1e-3 | row split | 44 | 8 | 12 | 16 | 20 | 1.000 |
| Looped 4x8, 1e-3 | row split | 45 | 8 | 16 | 20 | 28 | 1.000 |
| Looped 4x8, 1e-3 | row split | 46 | 8 | 12 | -- | 32 | 0.650 |
| Looped 4x8, 1e-3 | prompt-disjoint | 42 | 8 | 11 | 15 | 16 | 1.000 |
| Untied, 1e-3 | row split | 42 | 27 | -- | -- | -- | 0.073 |
| Untied, 1e-3 | row split | 43 | -- | -- | -- | -- | 0.074 |
| Untied, 1e-3 | row split | 44 | 31 | -- | -- | -- | 0.083 |
| Untied, 1e-3 | row split | 45 | 27 | -- | -- | -- | 0.087 |
| Untied, 1e-3 | row split | 46 | 30 | -- | -- | -- | 0.165 |
| Untied, 3e-4 | prompt-disjoint | 42 | 26 | 28 | 29 | 30 | 1.000 |
| Untied, 3e-4 | prompt-disjoint | 43 | 17 | 25 | 29 | 30 | 1.000 |
| Untied, 3e-4 | prompt-disjoint | 44 | 25 | 28 | 32 | 31 | 0.982 |
| Untied + aux., 1e-3 | row split | 44 | 16 | 17 | 20 | 28 | 0.855 |
| Untied + aux., 1e-3 | row split | 45 | 10 | 17 | 22 | 24 | 1.000 |
| Looped 4x8, 3e-4 | prompt-disjoint | 42 | 7 | 12 | 16 | 20 | 1.000 |
| Looped 4x8, 3e-4 | prompt-disjoint | 43 | 8 | 13 | 20 | 24 | 1.000 |
| Looped 4x8, 3e-4 | prompt-disjoint | 44 | 7 | 10 | 12 | 16 | 1.000 |
