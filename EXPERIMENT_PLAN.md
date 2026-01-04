# HoloGrad 논문 검증 실험 계획

## 1. 검증할 논문 주장

| ID | 주장 | 기대값 | 논문 섹션 |
|----|------|--------|----------|
| C1 | Random projection은 K/D ≥ 0.01 필요 | cosine ≈ √(K/D) | 6.2 |
| C2 | NN gradient는 batch간 거의 직교 | cosine ≈ 0.07 | 6.1 |
| C3 | Momentum이 random보다 180× 효율적 | 0.18 vs 0.001 alignment | 7 |
| C4 | Momentum-Centric = SGD의 30-40% | loss 감소율 비교 | 7 |
| C5 | ADC가 gradient subspace 학습 | γ_t 시간에 따라 증가 | 5 |
| C6 | Scale-corrected estimator는 unbiased | E[ĝ] = g | Thm 1 |
| C7 | Trimmed mean이 Byzantine 20% 제거 | τ=0.1에서 tolerance | 8 |
| C8 | PoGP scalar는 재현 가능 | \|a-a*\| ≤ ε | 4.4 |

## 2. 실험 목록

| # | 실험명 | 검증 주장 | 모델 | 예상 시간 |
|---|--------|----------|------|----------|
| E1 | Gradient Variability | C2 | GPT2-like (50M) | 1h |
| E2 | K/D Ratio Sweep | C1 | Tiny + Small | 3h |
| E3 | Momentum vs Random | C3, C4 | Small (10M) | 2h |
| E4 | ADC Captured Energy | C5 | Tiny (500K) | 1h |
| E5 | Unbiased Estimator | C6 | Tiny (500K) | 0.5h |
| E6 | PoGP Verification | C8 | Tiny (500K) | 0.5h |
| E7 | Byzantine Tolerance | C7 | Tiny (500K) | 1h |

**총 예상: ~9시간**

## 3. 모델 설정

| 이름 | n_layer | n_head | n_embd | D (params) | 용도 |
|------|---------|--------|--------|------------|------|
| Tiny | 2 | 2 | 64 | ~500K | E2,E4,E5,E6,E7 |
| Small | 4 | 4 | 256 | ~10M | E2,E3 |
| GPT2-like | 6 | 8 | 512 | ~50M | E1 |

## 4. 구현 상태

### 완료
- `worker_server.py` - OOM 방지, 안정성, 메모리 관리
- `coordinator_server.py` - 재시도 로직, 체크포인트, 워커 실패 복구
- `deploy.py` - Vast.ai 원클릭 배포
- `DirectionGenerator` - random unit direction 생성
- `ADCCodebook` - Streaming Oja-QR 코드북 학습
- `RobustAggregator` - trimmed mean 집계
- JVP gradient projection 계산

### 미구현 (필요)
| # | 작업 | 용도 | 우선순위 |
|---|------|------|---------|
| W1 | Gradient variability 측정 스크립트 | E1 | ⭐⭐⭐ |
| W2 | Momentum-Centric coordinator/worker | E3 | ⭐⭐⭐ |
| W3 | Full SGD baseline trainer | E3 비교 | ⭐⭐ |
| W4 | Byzantine worker injection | E7 | ⭐⭐ |
| W5 | 실험 자동화 스크립트 | 전체 | ⭐⭐⭐ |

## 5. Vast.ai 인프라

### 명령어
```bash
# 인스턴스 검색 (RTX 4090, 가격순)
vastai search offers 'gpu_name=RTX_4090 rentable=true' -o 'dph'

# 인스턴스 생성
vastai create instance {offer_id} \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --disk 50

# 인스턴스 조회
vastai show instances

# 인스턴스 종료
vastai destroy instance {id}

# 전체 종료
vastai show instances --raw | jq -r '.[].id' | xargs -I {} vastai destroy instance {}
```

### 배포/실행
```bash
# 워커 배포
python scripts/distributed/deploy.py --deploy

# 학습 시작
python scripts/distributed/deploy.py --train --steps 1000 --K 512 --lr 0.01

# 상태 확인
python scripts/distributed/deploy.py --status

# 배포 + 학습
python scripts/distributed/deploy.py --deploy --train
```

### 비용
- RTX 4090: ~$0.25/hr
- 10개 × 10시간 = $25
- 여유 포함: ~$30-40 권장

## 6. 실행 순서

### Phase 1: 구현 (로컬)
1. W1: Gradient variability 측정 스크립트
2. W2: Momentum-Centric 구현
3. W5: 실험 자동화 스크립트
4. git push

### Phase 2: 인프라 (Vast.ai)
1. RTX 4090 인스턴스 10개 생성
2. `deploy.py --deploy` 실행
3. 워커 헬스체크 확인

### Phase 3: 기본 검증
1. E1: Gradient variability (논문 핵심 관찰)
2. E5: Unbiased estimator
3. E6: PoGP verification

### Phase 4: 핵심 실험
1. E2: K/D ratio sweep
2. E3: Momentum vs Random
3. E4: ADC captured energy

### Phase 5: 시스템 실험
1. E7: Byzantine tolerance

### Phase 6: 정리
1. 인스턴스 종료
2. 결과 시각화
3. 논문 claim 검증 정리

## 7. 실험 상세

### E1: Gradient Variability
```
목적: "pairwise cosine similarity ≈ 0.07" 검증
모델: GPT2-like (50M params)
데이터: WikiText-2
방법:
  1. 50개 mini-batch에서 full gradient 계산
  2. pairwise cosine similarity 계산
  3. leave-one-out SVD reconstruction
측정:
  - pairwise cosine similarity
  - gradient variance ratio
예상: cosine ≈ 0.07, variance ratio ≈ 0.93
```

### E2: K/D Ratio Sweep
```
목적: "K/D ≥ 0.01이어야 수렴" 검증
모델: Tiny (500K), Small (10M)
K: [8, 16, 32, 64, 128, 256, 512]
측정:
  - Loss 감소율
  - Reconstructed vs true gradient cosine
예상:
  - K/D < 0.01: 수렴 안됨
  - K/D > 0.01: 수렴 시작
```

### E3: Momentum vs Random
```
목적: "Momentum 180× 효율적" 검증
모델: Small (10M)
비교:
  A) Random K=64 (64 scalars)
  B) Random K=1 (1 scalar)
  C) Momentum K=1 (1 scalar, momentum direction)
  D) Full SGD (baseline)
측정:
  - Loss 감소 (같은 step)
  - 통신량 대비 효율
예상: Momentum >> Random K=1, Momentum ≈ Random K=64
```

### E4: ADC Captured Energy
```
목적: "ADC가 subspace 학습" 검증
모델: Tiny (500K)
ADC rank: [8, 16, 32, 64]
측정:
  - γ_t = ||UU^T g||² / ||g||² over time
  - Loss 감소 (ADC on vs off)
예상: γ_t 증가, ADC on > ADC off
```

### E5: Unbiased Estimator
```
목적: "E[ĝ] = g" 검증
방법:
  1. 고정된 θ에서 true gradient g 계산
  2. K개 random direction으로 ĝ 계산
  3. 100회 반복하여 E[ĝ] 계산
측정: ||E[ĝ] - g|| / ||g||
예상: < 0.01 (unbiased)
```

### E6: PoGP Verification
```
목적: "scalar 재현 가능" 검증
방법:
  1. Worker: (θ, B, s) → a 계산
  2. Verifier: 같은 입력 → a* 계산
  3. |a - a*| 분포 측정
측정: |a - a*| 분포, 적절한 ε
예상: deterministic 환경에서 |a - a*| ≈ 0
```

### E7: Byzantine Tolerance
```
목적: "trimmed mean이 20% 악의적 워커 제거" 검증
설정:
  - 악의적 비율: [0%, 10%, 20%, 30%]
  - 공격: random scalar, sign-flip, extreme
  - τ = 0.1 (10% trim)
측정: Loss 감소 (with/without trimming)
예상: 20%까지 tolerance, 30%에서 실패
```

## 8. 예상 산출물

```
results/
├── figures/
│   ├── gradient_variability.png
│   ├── kd_ratio_sweep.png
│   ├── momentum_vs_random.png
│   ├── adc_captured_energy.png
│   └── byzantine_tolerance.png
├── data/
│   ├── e1_gradient_variability.json
│   ├── e2_kd_ratio.json
│   ├── e3_momentum_comparison.json
│   ├── e4_adc_energy.json
│   └── e7_byzantine.json
└── RESULTS_SUMMARY.md
```

## 9. 리스크

| 리스크 | 확률 | 대응 |
|--------|------|------|
| K/D 법칙 불일치 | 낮음 | 이론/구현 재검토 |
| Momentum 효과 미미 | 중간 | 하이퍼파라미터 튜닝 |
| OOM | 중간 | 모델/배치 축소 |
| 네트워크 지연 | 높음 | timeout 증가, 재시도 |
| 인스턴스 불안정 | 중간 | 체크포인트 자주 저장 |
