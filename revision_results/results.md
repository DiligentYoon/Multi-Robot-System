# 기존 로그 평가 결과

- 코드 / 데이터 기준 commit: `63be0aabd76554a363a8f871a831c85eed580ded`
- 저장 episode: 480; 통계에 포함한 episode: 336.
- CMR: main_driver의 any-agent barrier < -1e-3 기준 유지. Graph Connectivity 지표 추가 없음.
- 연속 지표: episode 평균 → seed 평균 → seed 간 평균/표준편차. Runtime pooled p95는 agent 중복 없이 실제 batch sample에서 계산.
- 결정론적 Frontier: position / nominal-control trajectory가 완전히 동일한 반복만 map당 최소 seed 1회로 집계.
- SR Wilson interval은 episode count 기반 기술통계. 동일 map의 반복 seed를 독립 map 일반화 증거로 해석하지 않음.

| 방법 | N | 성공 / 평가 | SR (%) | 95% CI (%) | 장애물 충돌 | Agent 충돌 | Frozen | Timeout |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| w/o CM | 5 | 54/60 | 90.0 | 79.9-95.3 | 0 | 0 | 6 | 0 |
| Full | 3 | 57/60 | 95.0 | 86.3-98.3 | 0 | 0 | 3 | 0 |
| Full | 5 | 56/60 | 93.3 | 84.1-97.4 | 0 | 0 | 4 | 0 |
| Full | 7 | 58/60 | 96.7 | 88.6-99.1 | 1 | 0 | 1 | 0 |
| w/o ICA | 5 | 38/60 | 63.3 | 50.7-74.4 | 1 | 21 | 0 | 0 |
| Frontier | 3 | 4/12 | 33.3 | 13.8-60.9 | 0 | 0 | 8 | 0 |
| Frontier | 5 | 2/12 | 16.7 | 4.7-44.8 | 1 | 0 | 9 | 0 |
| Frontier | 7 | 5/12 | 41.7 | 19.3-68.0 | 0 | 0 | 7 | 0 |

| 방법 | N | CMR (%) | SPR (%) | 평균 MCM (m²) | 평균 위반 총시간 (s) | 평균 episode 최장 위반 (s) |
|---|---:|---:|---:|---:|---:|---:|
| w/o CM | 5 | 85.55 | 99.26 | -0.8458 | 37.38 | 14.21 |
| Full | 3 | 97.86 | 99.96 | -0.0154 | 7.22 | 2.08 |
| Full | 5 | 96.94 | 99.37 | -0.0282 | 8.92 | 2.77 |
| Full | 7 | 98.84 | 98.95 | 0.0652 | 4.00 | 1.14 |
| w/o ICA | 5 | 98.22 | 92.77 | 0.0770 | 3.27 | 1.43 |
| Frontier | 3 | 97.45 | 100.00 | 0.0542 | 5.35 | 2.32 |
| Frontier | 5 | 96.61 | 98.32 | -0.0968 | 8.33 | 2.62 |
| Frontier | 7 | 98.55 | 98.18 | 0.1569 | 1.58 | 1.52 |

| 방법 | N | 위반 episode / 평가 | 전체 최저 margin (m²) | 전체 최장 연속 위반 (s) |
|---|---:|---:|---:|---:|
| w/o CM | 5 | 37/60 | -6.3781 | 144.40 |
| Full | 3 | 33/60 | -0.8381 | 14.80 |
| Full | 5 | 28/60 | -1.2525 | 18.90 |
| Full | 7 | 18/60 | -0.5224 | 12.60 |
| w/o ICA | 5 | 21/60 | -1.2155 | 16.70 |
| Frontier | 3 | 3/12 | -0.6213 | 12.80 |
| Frontier | 5 | 5/12 | -1.0992 | 13.10 |
| Frontier | 7 | 2/12 | -0.0491 | 17.30 |

| 방법 | N | Batch mean (ms) | Batch p95 (ms) | Batch max (ms) | >100 ms (%) |
|---|---:|---:|---:|---:|---:|
| w/o CM | 5 | 13.53 | 25.01 | 130.32 | 0.013 |
| Full | 3 | 10.39 | 19.48 | 122.46 | 0.011 |
| Full | 5 | 13.24 | 23.01 | 123.71 | 0.015 |
| Full | 7 | 15.83 | 30.54 | 138.96 | 0.037 |
| w/o ICA | 5 | 11.62 | 20.21 | 192.69 | 0.002 |
| Frontier | 3 | 11.05 | 20.93 | 91.51 | 0.000 |
| Frontier | 5 | 13.96 | 26.22 | 116.51 | 0.032 |
| Frontier | 7 | 19.20 | 36.71 | 133.40 | 0.092 |

| 방법 | N | Goal 사건 수 | 첫 도착 시 목표 내부 비율 (%) | 평균 goal-region 거리 (m) | 최대 goal-region 거리 (m) |
|---|---:|---:|---:|---:|---:|
| w/o CM | 5 | 44 | 20.00 | 0.690 | 1.442 |
| Full | 3 | 47 | 34.00 | 0.375 | 0.724 |
| Full | 5 | 46 | 20.40 | 0.580 | 1.214 |
| Full | 7 | 48 | 14.29 | 0.713 | 1.409 |
| w/o ICA | 5 | 30 | 20.00 | 0.509 | 1.086 |
| Frontier | 3 | 4 | 33.33 | 0.296 | 0.670 |
| Frontier | 5 | 1 | 20.00 | 1.006 | 2.126 |
| Frontier | 7 | 4 | 14.29 | 0.522 | 1.073 |

## 검증 및 해석

- `goal_map_free_cell_count_mismatch`: 80 episodes. `data_audit.csv`에서 확인.
- `obs_avoid_viol` / `agent_avoid_viol` / `agent_conn_viol` residual은 CMR·SPR 판정에 사용하지 않았습니다.
- Runtime은 해당 run에서 저장된 batch 경로 시간입니다. 개별 agent QP 시간·전체 control loop 시간·현재 기기의 신규 측정값은 아닙니다. 당시 hardware 및 solver 버전은 로그에 없습니다.
- 위반 기간은 기록 step 수 × dt입니다. CSV time은 첫 post-step 상태에 0을 붙이므로 T×dt와 마지막 time은 dt만큼 다릅니다.
- Goal 상태는 성공 종료 episode에서만 집계하고, 성공 episode 평균 후 성공 사례가 있는 seed 간 균등 평균을 취했습니다. 유클리드 거리는 goal 셀 영역까지의 거리이며 경로 거리·last-agent 도달 시간을 의미하지 않습니다.
- 논문의 기존 MCM처럼 episode별 최소 margin을 평균했습니다. 단일 episode 최악값과 구분해야 합니다.

## 추가 기록 또는 실험이 필요한 항목

- 기존 로그에는 coordination planner/MST/router 시간, control compute 시간, active constraint 수, hardware/solver/run config가 없습니다. 다음 run부터 새 batch_log.csv·run_metadata.json에 기록하고 집계합니다. 전체 wall-clock loop에는 printing·figure·logging도 포함되므로 control_compute_s와 구분합니다.
- First-arrival 이후 team arrival / last-arrival time: 기존 로그는 즉시 종료되므로 Goal Rally 구현 및 rollout 필요.
- 강화 Frontier, noise, sensitivity의 신규 결과: 현재 데이터만으로 계산할 수 없음.
- Map obstacle fraction은 경계와 Square occupied background를 포함하는 raster 분율입니다. 국소 장애물 밀도나 통로 폭과 동일하지 않음.

## 출력 파일

- `per_episode.csv`: 저장된 모든 episode, 원래 summary 지표, 새 통계, primary 포함 여부.
- `group_summary.csv`: 방법·팀 크기를 구분한 최종 집계, SR CI, runtime pooled 통계.
- `by_map.csv`, `paired_full_frontier.csv`: map별 기술 비교.
- `deterministic_repeat_audit.csv`: 반복 trajectory 동일성 검증.
- `data_audit.csv`: 데이터 누락·불일치 목록.
- `map_manifest.csv`, `metadata.json`: map 대응과 분석 parameter·버전.
