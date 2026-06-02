# 18 — Wave estimator end-to-end (no oracle)

```
Train: 114 forms / 1928 points
Holdout: 47 forms / 788 points
Conformal q_0.95: B(detected)=6.00  C(oracle)=6.96

=== GLOBAL (holdout) ===
Method                 n  cumMAPE  incSMAPE  cov95  cov+cf  Winkler  width    bias
A prod               788    20.0%     0.667    79%       -       14      9       0
B wave_detected      688    13.0%     0.667    80%     96%        9      5       0
C wave_oracle        552    18.2%     0.667    77%     95%       20      8      -1
D naive_recent       788    25.0%     0.667      -       -        -      -       1
E naive_overall      788    20.0%     0.667      -       -        -      -       0

DETECTION PENALTY (paired n=492): detected=16.7%  oracle=18.2%  penalty=+-1.5pp

=== SIGNIFICANCE (Wilcoxon signed-rank, paired cum APE) ===
  B detected vs D naive_recent: n=688 med(b_detected)=13.0% med(d_naive_recent)=25.1% p=0.0000 **SIG** winner=b_detected
  B detected vs A prod: n=688 med(b_detected)=13.0% med(a_prod)=20.0% p=0.0000 **SIG** winner=b_detected
  C oracle vs B detected (penalty): n=492 med(c_oracle)=18.2% med(b_detected)=16.7% p=0.0851 ns winner=b_detected
  D naive_recent vs A prod: n=788 med(d_naive_recent)=25.0% med(a_prod)=20.0% p=0.0000 **SIG** winner=a_prod

=== PER TYPE (cum MAPE, holdout) ===
ftype                    A prod    B det   C orac  D naive     n
creative_submission          9%       0%       9%      20%    40
event_feedback              52%      15%      50%      37%    32
event_registration          17%      11%      17%      20%   200
holiday                     31%      17%      23%      35%    32
other                       27%      13%      17%      19%    40
political                   18%      14%      25%      10%    20
recruitment                  7%       8%       8%      13%    40
service                     21%      29%      29%      24%    40
survey                      73%      47%      39%      66%   108
unknown                     23%      13%      17%      22%   156
volunteer_donor             13%       6%      12%      26%    80

=== PER HORIZON (cum MAPE, holdout) ===
horizon_h      A prod    B det   C orac  D naive     n
0.5               17%       7%      11%      12%   197
1.0               17%      12%      17%      15%   197
2.0               23%      14%      20%      27%   197
6.0               29%      21%      30%      67%   197
```
