# Table 1: looped 4x8 against untied 32-layer

Seeds that solve all 15 hops at N=16 (H* = 15 at the final evaluation), and H* per seed at N=32.

| Task | lr | Looped 4x8 | Untied 32-layer | Paper (looped / untied) |
|---|---|---|---|---|
| N=16 | 1e-3 | 5/5 | 0/5 | 5/5 / 0/5 |
| N=16 | 3e-4 | 5/5 | 4/5 | 5/5 / 4/5 |
| N=16 | 1e-4 | 1/5 | 1/5 | 1/5 / 1/5 |
| N=16, untied 16 layers | 3e-4 | -- | 1/5 | -- / 1/5 |
| N=16, auxiliary loss | 1e-3 | -- | 2/5 | -- / 2/5 |
| N=32 | 3e-4 | H* = 6, 0, 1 | H* = 1, 1, 1 | H* = 6, 0, 1 / H* = 1, 1, 1 |

# Numbers in the text

| Quantity | Value | Paper |
|---|---|---|
| Mean first-solve step at lr 3e-4, looped 4x8 (solving seeds) | 44.8k over 5 | 45k |
| Mean first-solve step at lr 3e-4, untied 32-layer (solving seeds) | 55.0k over 4 | 55k |
| Looped 4x8 without tau_t, lr 1e-3, seeds that solve | 3/4 | three of four |
| Untied 4-layer at N=32, lr 3e-4, H* per seed (42, 43, 46, 47) | 1, 1, 0, 1 | at most 1, four seeds |

# Every seed

| Condition | Run | H* (final) | First step with H* = 15 |
|---|---|---|---|
| N16, lr 1e-3, looped | ouro-4x8_s42 | 15 | 72000 |
| N16, lr 1e-3, looped | ouro-4x8_s43 | 15 | 88000 |
| N16, lr 1e-3, looped | ouro-4x8_s44 | 15 | 56000 |
| N16, lr 1e-3, looped | ouro-4x8_s45 | 15 | 64000 |
| N16, lr 1e-3, looped | ouro-4x8_s46 | 15 | 96000 |
| N16, lr 1e-3, untied | deep-32_s42 | 1 | -- |
| N16, lr 1e-3, untied | deep-32_s43 | 1 | -- |
| N16, lr 1e-3, untied | deep-32_s44 | 2 | -- |
| N16, lr 1e-3, untied | deep-32_s45 | 3 | -- |
| N16, lr 1e-3, untied | deep-32_s46 | 2 | -- |
| N16, lr 3e-4, looped | ouro-4x8-lr3e4_s42 | 15 | 44000 |
| N16, lr 3e-4, looped | ouro-4x8-lr3e4_s43 | 15 | 48000 |
| N16, lr 3e-4, looped | ouro-4x8-lr3e4_s44 | 15 | 48000 |
| N16, lr 3e-4, looped | ouro-4x8-lr3e4_s45 | 15 | 40000 |
| N16, lr 3e-4, looped | ouro-4x8-lr3e4_s46 | 15 | 44000 |
| N16, lr 3e-4, untied | deep-32-lr3e4_s42 | 15 | 52000 |
| N16, lr 3e-4, untied | deep-32-lr3e4_s43 | 15 | 64000 |
| N16, lr 3e-4, untied | deep-32-lr3e4_s44 | 15 | 56000 |
| N16, lr 3e-4, untied | deep-32-lr3e4_s45 | 15 | 48000 |
| N16, lr 3e-4, untied | deep-32-lr3e4_s46 | 6 | -- |
| N16, lr 1e-4, looped | ouro-4x8-lr1e4_s42 | 7 | -- |
| N16, lr 1e-4, looped | ouro-4x8-lr1e4_s43 | 7 | -- |
| N16, lr 1e-4, looped | ouro-4x8-lr1e4_s44 | 5 | -- |
| N16, lr 1e-4, looped | ouro-4x8-lr1e4_s45 | 8 | -- |
| N16, lr 1e-4, looped | ouro-4x8-lr1e4_s46 | 15 | 92000 |
| N16, lr 1e-4, untied | deep-32-lr1e4_s42 | 4 | -- |
| N16, lr 1e-4, untied | deep-32-lr1e4_s43 | 4 | -- |
| N16, lr 1e-4, untied | deep-32-lr1e4_s44 | 15 | 96000 |
| N16, lr 1e-4, untied | deep-32-lr1e4_s45 | 2 | -- |
| N16, lr 1e-4, untied | deep-32-lr1e4_s46 | 2 | -- |
| N16 untied 16 layers, lr 3e-4, untied | deep-16-lr3e4_s42 | 3 | -- |
| N16 untied 16 layers, lr 3e-4, untied | deep-16-lr3e4_s43 | 15 | 76000 |
| N16 untied 16 layers, lr 3e-4, untied | deep-16-lr3e4_s44 | 3 | -- |
| N16 untied 16 layers, lr 3e-4, untied | deep-16-lr3e4_s45 | 6 | -- |
| N16 untied 16 layers, lr 3e-4, untied | deep-16-lr3e4_s46 | 3 | -- |
| N16 auxiliary loss, lr 1e-3, untied | deep-32-aux_s42 | 1 | -- |
| N16 auxiliary loss, lr 1e-3, untied | deep-32-aux_s43 | 4 | -- |
| N16 auxiliary loss, lr 1e-3, untied | deep-32-aux_s44 | 15 | 96000 |
| N16 auxiliary loss, lr 1e-3, untied | deep-32-aux_s45 | 15 | 80000 |
| N16 auxiliary loss, lr 1e-3, untied | deep-32-aux_s46 | 7 | -- |
| N16 no tau_t, lr 1e-3, looped | ouro-4x8-notau_s42 | 15 | 68000 |
| N16 no tau_t, lr 1e-3, looped | ouro-4x8-notau_s43 | 15 | 80000 |
| N16 no tau_t, lr 1e-3, looped | ouro-4x8-notau_s44 | 15 | 56000 |
| N16 no tau_t, lr 1e-3, looped | ouro-4x8-notau_s45 | 9 | -- |
| N32, lr 3e-4, looped | ouro-4x8_n32_s42 | 6 | -- |
| N32, lr 3e-4, looped | ouro-4x8_n32_s43 | 0 | -- |
| N32, lr 3e-4, looped | ouro-4x8_n32_s44 | 1 | -- |
| N32, lr 3e-4, untied | deep-32_n32_s42 | 1 | -- |
| N32, lr 3e-4, untied | deep-32_n32_s43 | 1 | -- |
| N32, lr 3e-4, untied | deep-32_n32_s44 | 1 | -- |
| N32 untied 4 layers, lr 3e-4, untied | deep-4_n32_s42 | 1 | -- |
| N32 untied 4 layers, lr 3e-4, untied | deep-4_n32_s43 | 1 | -- |
| N32 untied 4 layers, lr 3e-4, untied | deep-4_n32_s46 | 0 | -- |
| N32 untied 4 layers, lr 3e-4, untied | deep-4_n32_s47 | 1 | -- |
