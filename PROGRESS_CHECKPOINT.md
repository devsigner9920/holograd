# HoloGrad 프로젝트 진행 체크포인트

**마지막 업데이트**: 2026-01-02
**현재 상태**: 논문 수정 및 질문 분석 완료

---

## 0. 최근 세션 완료 작업 (2026-01-02)

### 논문 수정 (holograd_tex.tex)
1. **ADC 초기화/Bootstrap 섹션 추가** (Section 5.2)
   - Cold-start problem 설명
   - Random-direction warmup vs Bootstrap 전략 비교
   - Verifiability 영향 분석

2. **Variance Analysis 수정** (Section 4.3)
   - ADC scale_factor = rank (not 1) 명시
   - Normalized subspace direction 설명
   - Variance correction 수식 추가: `sqrt(K/effective_dim)`

3. **실험 조건 명확화** (Section 6.2)
   - K/D 비율 테이블 추가
   - Practical threshold: K/D ≳ 0.01

### 질문 분석 결과
1. **Bootstrap vs Verifiability**: Bootstrap은 NOT verifiable, 이후 HoloGrad는 fully verifiable ("Partial verifiability")
2. **Momentum-centric without bootstrap**: ✅ 가능 - warmup 동안 random directions로 momentum 축적
3. **Dynamic ADC rank**: 가능하나 미구현 - captured_energy_ratio 기반 조절 전략 제안

---

## 1. 완료된 작업

### 1.1 Variance 문제 수정
- **문제**: ADC scale_factor가 `dimension` (1.6M)을 반환하여 variance가 폭발
- **해결**: `rank` (16~32)를 반환하도록 수정
- **위치**: `src/holograd/protocol/direction.py` line 244-247

### 1.2 Aggregation Variance Correction
- **추가**: `sqrt(K/effective_dim)` variance correction
- **ADC**: effective_dim = rank
- **Random**: effective_dim = dimension
- **위치**: `src/holograd/protocol/aggregation.py`

### 1.3 ADC Warmup 문제 수정 (핵심)
- **문제**: ADC warmup 동안에도 ADC direction 사용 → chicken-egg 문제
  - Random ADC subspace는 gradient의 0.016%만 포착
  - 재구성된 gradient가 거의 noise → ADC 초기화도 실패
- **해결**: ADC warmup 동안 random directions 사용
- **위치**: `src/holograd/distributed/coordinator.py`

### 1.4 Bootstrap ADC 기능 추가
- **기능**: `trainer.bootstrap_adc(num_steps, lr)` 메서드 추가
- **동작**: True gradient (autograd)로 ADC subspace 초기화
- **필요성**: 고차원 문제에서 ADC가 올바른 subspace를 학습하려면 필수
- **위치**: `src/holograd/training/trainer.py`

---

## 2. 검증 결과

### 2.1 Quadratic Benchmark (2D)
| Variant | Final Loss | 상태 |
|---------|-----------|------|
| Full-Space Random | ~1.60 | ✅ |
| ADC HoloGrad (r=32) | ~1.74 | ✅ |

### 2.2 WikiText Training (1.6M params)
| 방식 | 시작 Loss | 최종 Loss | 상태 |
|-----|----------|----------|------|
| HoloGrad (bootstrap 없이) | 10.83 | 10.83 | ❌ 학습 안됨 |
| Bootstrap (20 steps) + HoloGrad (30 steps) | 10.80 | **9.30** | ✅ |

### 2.3 테스트
- 139개 전체 테스트 통과 ✅

---

## 3. 핵심 발견사항

### 3.1 HoloGrad의 근본적 한계
Random direction 기반 gradient 재구성은 **K ≈ D** 일 때만 잘 작동함:

| 문제 | D (params) | K (directions) | K/D | 결과 |
|-----|-----------|----------------|-----|------|
| Quadratic | 2 | 16 | 8.0 | ✅ 작동 |
| WikiText | 1.6M | 30 | 0.00002 | ❌ 실패 |

### 3.2 ADC의 필요성
- 고차원 문제에서 ADC (Adaptive Direction Codebook)는 **필수**
- ADC는 gradient가 집중되는 low-rank subspace를 학습
- 단, ADC 초기화에 **true gradient가 필요** (bootstrap 단계)

### 3.3 Bootstrap 전략
1. 처음 N steps: autograd로 true gradient 계산 → ADC 초기화
2. 이후: HoloGrad protocol로 학습 계속
3. Trade-off: Bootstrap 비용 vs 통신 효율성

---

## 4. 논문 수정 필요 사항 ✅ 완료

### 4.1 ADC 초기화 섹션 추가 ✅
- ~~Random ADC subspace로는 고차원 문제에서 작동하지 않음~~
- ~~Bootstrap 단계 필요성 명시~~
- ~~또는 momentum-centric 방식 사용~~
- **→ Section 5.2 "ADC Initialization and the Cold-Start Problem" 추가됨**

### 4.2 Variance Analysis 수정 ✅
- ~~ADC scale_factor = rank (not dimension)~~
- ~~Variance correction: `sqrt(K/effective_dim)`~~
- **→ Section 4.3 수정됨: normalized subspace direction, scale_factor = r**

### 4.3 실험 조건 명확화 ✅
- ~~Quadratic (저차원) vs WikiText (고차원) 차이 설명~~
- ~~ADC warmup 전략 비교 실험 필요~~
- **→ Section 6.2 K/D 비율 테이블 및 practical threshold 추가됨**

---

## 5. 추후 작업 (클라우드 환경 필요)

### 5.1 대규모 실험
- [ ] WikiText 전체 학습 (500+ steps)
- [ ] GPT-2 Small (124M params) 학습
- [ ] 다양한 K, rank 조합 실험

### 5.2 분산 환경 테스트
- [ ] 실제 multi-node 환경에서 통신 효율성 측정
- [ ] Byzantine worker 시뮬레이션
- [ ] Verification protocol 성능 측정

### 5.3 비교 실험
- [ ] HoloGrad vs Standard SGD (통신 비용 포함)
- [ ] HoloGrad vs SignSGD
- [ ] HoloGrad vs FedAvg

### 5.4 Ablation Studies
- [ ] Bootstrap steps 수 vs 최종 성능
- [ ] ADC rank vs 수렴 속도
- [ ] Warmup 전략 비교 (random vs true gradient)

---

## 6. 파일 구조 요약

```
src/holograd/
├── protocol/
│   ├── direction.py      # ADC scale_factor 수정
│   └── aggregation.py    # Variance correction 추가
├── distributed/
│   └── coordinator.py    # ADC warmup 로직 수정
└── training/
    └── trainer.py        # bootstrap_adc() 추가
```

---

## 7. 명령어 참고

```bash
# 테스트 실행
PYTHONPATH=src pytest tests/ -v

# Quadratic benchmark
python benchmarks/compare_variants_quadratic.py

# WikiText with bootstrap
python -c "
from holograd.training.trainer import HoloGradTrainer
# ... (setup)
trainer.bootstrap_adc(num_steps=20, lr=0.1)  # True gradient로 ADC 초기화
trainer.train(num_steps=100)  # HoloGrad로 계속 학습
"
```

---

## 8. 질문/이슈 ✅ 분석 완료

### Q1: Bootstrap 단계가 논문의 "verifiable" 특성에 영향을 주는가? ✅
**답변**: Yes, 영향 있음 ("Partial verifiability")
- Bootstrap 단계: true gradient 사용 → NOT verifiable (coordinator 신뢰 필요)
- 이후 HoloGrad: scalar PoGP proofs → fully verifiable
- 일회성 초기화 비용이므로 수용 가능한 trade-off

### Q2: Momentum-centric 모드가 bootstrap 없이 작동할 수 있는가? ✅
**답변**: Yes, 가능함
- Warmup 동안 random directions 사용 (coordinator.py line 99-107)
- Noisy하지만 unbiased gradient 추정 가능
- Momentum = reconstructed gradient의 EMA로 축적
- ADC와 달리 subspace 학습 불필요 → bootstrap 불필요

### Q3: ADC rank를 동적으로 조절하는 것이 가능한가? ✅
**답변**: 이론적으로 가능, 현재 미구현
- 지표: `captured_energy_ratio(gradient)` 활용 가능
- 전략: γ < 0.5이면 rank 증가, γ ≈ 1이면 rank 감소
- 구현 시 고려사항: codebook reshape, 통신 프로토콜 변경, hysteresis
