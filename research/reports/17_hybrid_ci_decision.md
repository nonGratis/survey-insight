# 17 - Hybrid CI decision rules (post-hoc analysis)

**Generated:** 2026-05-30T18:02:31
**Source:** 16_ab_points.csv (3710 backtest points, 3710 valid for hybrid)

## Методи

- **A · prod** — поточний (NHPP + P7 + P10 + P11).
- **B · selector_delta** — selector обирає модель, CI через delta-method.
- **C · oracle** — per-point вибір кращого Winkler (theoretical max).
- **D1 · rule_min_width** — use B if b_width <= a_width. Parameter-free.
- **D2 · rule_capped** — D1 + sanity cap: b_width <= max(20, 5 * point).
- **D3 · rule_horizon** — use B only if horizon <= 12h AND model in {asympexp, logistic}.

## Global summary

method|n|winkler_p50|winkler_p90|winkler_mean|width_p50|width_p90|width_p99|coverage_pct
a_prod|3710|234.0|2011.0|2675.4|158.0|656.0|1529.6|81.6
b_selector_delta|3710|289.5|963228.0|2528196.6|69.0|963228.0|35482648.0|79.3
c_oracle|3710|92.0|1212.0|2465.9|55.0|300.0|1179.9|80.5
d1_min_width|3710|116.0|2266.3|2630.8|44.0|256.0|977.6|65.6
d2_capped|3710|125.0|2266.3|2644.6|45.0|300.0|1062.0|65.0
d3_horizon|3710|206.5|4322.1|210391.1|143.0|911.0|337824.9|81.4


## Choice rates (яка частка пунктів обрала B)

rule|chose_b_pct|n
c_oracle|45.8|3710
d1_min_width|59.0|3710
d2_capped|54.3|3710
d3_horizon|29.4|3710


## Per horizon

horizon_h|method|n|winkler_p50|width_p50|width_p90|coverage_pct
2.0|a_prod|742|212.5|158.5|738.9|83.2
2.0|b_selector_delta|742|81.0|28.0|421035.4|86.8
2.0|c_oracle|742|46.0|30.0|305.2|83.8
2.0|d1_min_width|742|51.5|24.0|271.7|75.7
2.0|d2_capped|742|54.5|24.0|375.8|74.4
2.0|d3_horizon|742|132.0|89.0|3903.2|85.2
6.0|a_prod|742|220.5|158.5|738.9|84.1
6.0|b_selector_delta|742|109.0|28.0|421035.4|82.9
6.0|c_oracle|742|51.0|39.0|314.8|83.6
6.0|d1_min_width|742|71.5|24.0|271.7|71.6
6.0|d2_capped|742|79.5|24.0|375.8|70.5
6.0|d3_horizon|742|155.0|89.0|3903.2|81.1
24.0|a_prod|742|235.5|158.5|738.9|84.0
24.0|b_selector_delta|742|205.5|28.0|421035.4|73.9
24.0|c_oracle|742|83.0|49.5|332.9|80.1
24.0|d1_min_width|742|120.0|24.0|271.7|61.3
24.0|d2_capped|742|132.0|24.0|375.8|60.9
24.0|d3_horizon|742|235.5|158.5|738.9|84.0
72.0|a_prod|742|215.0|157.0|552.3|80.9
72.0|b_selector_delta|742|720.0|150.0|2727281.8|77.1
72.0|c_oracle|742|117.5|87.0|262.9|79.4
72.0|d1_min_width|742|157.0|55.0|254.0|62.9
72.0|d2_capped|742|159.0|55.0|254.0|62.9
72.0|d3_horizon|742|215.0|157.0|552.3|80.9
168.0|a_prod|742|256.0|160.0|555.9|76.0
168.0|b_selector_delta|742|979.5|282.5|2867748.3|75.7
168.0|c_oracle|742|159.0|105.0|300.0|75.7
168.0|d1_min_width|742|209.0|62.0|254.0|56.3
168.0|d2_capped|742|221.0|64.5|256.0|56.2
168.0|d3_horizon|742|256.0|160.0|555.9|76.0


## Per form_type (тільки A vs B vs D2)

form_type|method|n|winkler_p50|width_p50|coverage_pct
creative_submission|a_prod|210|116.0|99.0|92.9
creative_submission|b_selector_delta|210|81.5|7.5|69.5
creative_submission|d2_capped|210|42.0|7.0|65.2
event_feedback|a_prod|185|160.0|134.0|85.9
event_feedback|b_selector_delta|185|1965.0|1965.0|87.6
event_feedback|d2_capped|185|133.0|100.0|73.5
event_registration|a_prod|945|181.0|156.0|89.9
event_registration|b_selector_delta|945|122.0|19.0|72.6
event_registration|d2_capped|945|85.0|18.0|65.2
holiday|a_prod|205|419.0|256.0|77.6
holiday|b_selector_delta|205|725.0|576.0|94.6
holiday|d2_capped|205|160.0|107.0|82.4
other|a_prod|155|105.0|105.0|96.1
other|b_selector_delta|155|125.0|24.0|73.5
other|d2_capped|155|94.0|24.0|74.8
political|a_prod|100|218.0|160.5|81.0
political|b_selector_delta|100|172.0|77.0|87.0
political|d2_capped|100|105.0|42.0|71.0
recruitment|a_prod|235|320.0|274.0|83.0
recruitment|b_selector_delta|235|102.0|21.0|71.5
recruitment|d2_capped|235|80.0|19.0|62.1
service|a_prod|200|339.0|123.0|68.0
service|b_selector_delta|200|203.0|11.0|78.0
service|d2_capped|200|120.0|11.0|53.0
survey|a_prod|550|922.0|209.0|57.8
survey|b_selector_delta|550|68278.0|68278.0|90.4
survey|d2_capped|550|866.5|157.0|51.1
unknown|a_prod|570|231.5|157.5|83.0
unknown|b_selector_delta|570|288.0|96.0|81.9
unknown|d2_capped|570|88.0|51.0|68.6
volunteer_donor|a_prod|355|160.0|108.0|88.2
volunteer_donor|b_selector_delta|355|210.0|23.0|74.4
volunteer_donor|d2_capped|355|94.0|22.0|68.2


## Figures

- [Winkler per horizon — все методи](figures\17_winkler_per_horizon.html)
- [Width per horizon](figures\17_width_per_horizon.html)

## Інтерпретація

1. **C oracle** — upper bound. Якщо D-rules близькі — вони добре aproximate
   ідеальне рішення.
2. **D2 capped** — найбільш прод-realistic кандидат. Не вимагає знання
   winkler. Прозоре правило: "якщо delta-CI вузьке і не вибухає — використати".
3. **D3 horizon-based** — найкондервативніше, легко зрозуміти і пояснити.

Якщо D2 winkler_p50 ≤ A winkler_p50 І width значно менший → це winning
prod-кандидат. Промочуємо як P12.
