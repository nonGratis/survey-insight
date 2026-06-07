# 19 — Wave CI method A/B

```
=== CI METHOD A/B (holdout, target coverage 95%) ===
method             coverage  Winkler   relW  absW90  absWmax   >1x   >2x
M0 delta_raw          80.1%      9.0   0.42     104      380    0%    0%
M1 conf_delta         96.1%     18.0   1.33     358     1320   49%   30%
M2 ratio              75.1%     74.0   0.00     799     3060   43%   34%
M3 poisson            97.7%    108.0   6.93     259      484  100%   92%
M4 relative           96.7%     68.7   4.58     307     1059  100%  100%
M5 mondrian_rel       93.9%     28.6   2.53    1603     5535   59%   29%
M6 mondrian_delta     95.8%     18.4   1.35     500     1836   49%   31%
M7 mondrian_capped    85.8%     13.0   0.83     106      380    0%    0%

params: M1 conf_delta:q=6.00; M2 ratio:[0.00, 17.00]; M3 poisson:z=[-6.36, 27.89]; M4 relative:[-0.72, 4.58]; M5 mondrian_rel:edges=10/21 q={0: (-0.2, 3.3), 1: (-0.3, 1.5), 2: (-0.8, 26.9)}; M6 mondrian_delta:edges=10/21 q={0: 5.4, 1: 6.0, 2: 8.7}; M7 mondrian_capped:M6 + cap_width(max_rel=2.0)

None passed cap policy. Best covered by Winkler: M1 conf_delta cov=96% Winkler=18.0 cap>2x=30%

=== Coverage by truth-size bucket (key methods) ===
bucket        n       M1       M4       M5       M6
tiny<30     536      98%      98%      96%      98%
small        95      86%      93%      81%      86%
med          57      91%      88%      93%      93%

=== ANCHOR CASES (user screenshots): interval sanity ===
  47-form (1p0ERtAe): pred@6h=150 truth@6h=175 final=594
      M0 delta_raw       CI=[15, 300]  half=142
      M6 mondrian_delta  CI=[15, 1392]  half=688
      M7 mondrian_capped CI=[15, 300]  half=142
  7433-form (1GM-api8tg): pred@6h=152 truth@6h=6073 final=7433
      M0 delta_raw       CI=[15, 304]  half=144
      M6 mondrian_delta  CI=[15, 1411]  half=698
      M7 mondrian_capped CI=[15, 304]  half=144
```
