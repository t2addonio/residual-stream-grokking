# Experiment result summaries

Extracted from working-store archives on 2026-09-05. Raw `.npy` / `.npz` / logs remain in the tars.

## Dual interference — pure Dyck 80 seeds

n = 80
- mean `min_acc` = 0.0016
- min/max `min_acc` = 0.0000 / 0.0098
- seeds with `min_acc` < 0.05: **80/80 (100%)**
- mean final_acc = 0.9977
- mean min_correct_logit = -1.930

Every seed collapsed. Highest remaining accuracy after interference: seed 25 at 0.0098.

## Wider Dyck d_model=256, 40 seeds

n = 40
- mean `min_acc` = 0.0018
- min/max `min_acc` = 0.0000 / 0.0095
- seeds with `min_acc` < 0.05: **40/40 (100%)**
- mean final_acc = 0.9979
- mean min_correct_logit = -2.261

## Deep-wide Dyck 512×4, 40 seeds

n = 40
- mean `min_acc` = 0.0019
- min/max `min_acc` = 0.0000 / 0.0100
- seeds with `min_acc` < 0.05: **40/40 (100%)**
- mean final_acc = 0.9972
- mean min_correct_logit = -1.764

Note: an earlier writeup listed 39/40 at this scale; the extracted metas for this archive show all 40 seeds below 0.05 (worst seed 28 at 0.0100).

## GPT-2 Dyck-in-Text dual sweep, 24 seeds

n = 24
- mean `min_acc` = 0.3976
- min/max `min_acc` = 0.3667 / 0.4153
- seeds with `min_acc` < 0.05: **0/24 (0%)**
- mean final_acc = 0.5963

| seed | final_acc | min_acc | min_correct_logit |
|------|-----------|---------|-------------------|
| 0 | 0.5920 | 0.4060 | 0.571 |
| 1 | 0.6133 | 0.3940 | -7.540 |
| 2 | 0.5927 | 0.4067 | -1.905 |
| 3 | 0.6140 | 0.3860 | 22.479 |
| 4 | 0.6087 | 0.3913 | -9.171 |
| 5 | 0.5847 | 0.4153 | -4.558 |
| 6 | 0.5847 | 0.4147 | 20.255 |
| 7 | 0.6227 | 0.3667 | -5.919 |
| 8 | 0.5953 | 0.4013 | 7.690 |
| 9 | 0.6053 | 0.3873 | -5.990 |
| 10 | 0.5867 | 0.3927 | 17.439 |
| 11 | 0.5900 | 0.4093 | -59.690 |
| 12 | 0.6053 | 0.3947 | 2.737 |
| 13 | 0.6020 | 0.3960 | -0.396 |
| 14 | 0.5720 | 0.4000 | -2.444 |
| 15 | 0.6160 | 0.3813 | 11.702 |
| 16 | 0.5507 | 0.3913 | 6.846 |
| 17 | 0.5987 | 0.4000 | 3.511 |
| 18 | 0.6093 | 0.3907 | -36.228 |
| 19 | 0.5927 | 0.3993 | -1.132 |
| 20 | 0.5960 | 0.4040 | 35.103 |
| 21 | 0.6080 | 0.3920 | 15.756 |
| 22 | 0.5913 | 0.4087 | 9.300 |
| 23 | 0.5787 | 0.4140 | 5.625 |

## GPT-2 joint-4 simultaneous cancellation, 8 seeds

n = 8
- mean `best_acc` = 0.3952
- min/max `best_acc` = 0.3683 / 0.4167
- seeds with `best_acc` < 0.05: **0/8 (0%)**

| seed | best_acc | best_correct_logit |
|------|----------|--------------------|
| 0 | 0.4033 | 2.980 |
| 1 | 0.3950 | 11.953 |
| 2 | 0.4167 | 11.144 |
| 3 | 0.3800 | 2.507 |
| 4 | 0.3800 | -2.490 |
| 5 | 0.4133 | 8.263 |
| 6 | 0.4050 | 4.053 |
| 7 | 0.3683 | -1.051 |
