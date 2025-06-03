# 정보 검색 (Information Retrieval, IR) 학습

## 1. Learning to Rank (LTR)의 등장 배경 및 필요성

- **전통적인 랭킹 방법**:
    - BM25, Language Model (LM), Vector Space Model (VSM) 등은 개별적인 관련성 추정(Relevance Estimation) 방식입니다.
    - PageRank, HITS 등은 문서 자체의 중요도(Document Importance)를 평가합니다.
    - 이러한 개별적인 점수들을 어떻게 효과적으로 결합하여 최종 랭킹을 만들 것인가가 중요한 문제입니다.
- **단순 결합의 한계**:
    - **선형 결합**: $\alpha_1 \times \text{BM25} + \alpha_2 \times \text{LM} + \alpha_3 \times \text{PageRank} + \dots$
        - 각 가중치($\alpha_i$)를 어떻게 결정할 것인가? (수동 튜닝의 어려움)
    - **비선형 결합 (예: 의사결정 트리)**:
        - 어떤 기준으로 트리를 구성하고 분기할 것인가? (규칙 기반 접근의 복잡성)
- **수많은 특징(Features)의 등장**:
    - 쿼리-문서 간 매칭 점수, 문서의 정적/동적 특징, 사용자 행동 데이터 등 수천, 수만 가지의 특징을 활용할 수 있게 되면서, 이를 효과적으로 통합하여 랭킹 성능을 최적화할 필요성이 대두되었습니다.
- **LTR의 목표**: 다양한 특징(Relevance Estimators)들을 **자동으로 결합**하여 P@K, MAP, NDCG와 같은 **정보 검색(IR) 평가 지표를 직접 최적화**하는 모델을 학습하는 것입니다.

## 2. Learning to Rank의 기본 프레임워크

- **입력 (Input)**:
    - 각 (쿼리, 문서) 쌍은 여러 특징(Features)으로 표현된 벡터입니다. (예: `[BM25 점수, LM 점수, PageRank 점수, ...]`)
    - 각 문서에는 해당 쿼리에 대한 관련성 레이블(Relevance Label)이 주어집니다. (예: 0-관련 없음, 1-약간 관련, 2-매우 관련 등)
- **목표 (Objective)**: IR 평가 지표 (P@K, MAP, NDCG 등)를 최대화하는 것입니다.
- **출력 (Output)**: 랭킹 함수 $f(q, d) \rightarrow s$ (쿼리 $q$와 문서 $d$에 대한 관련성 점수 $s$를 출력). 이 점수를 기준으로 문서를 정렬합니다.
- **머신러닝 관점에서의 LTR**:
    - 주어진 학습 데이터 {(특징 벡터, 관련성 레이블)}를 이용하여, 평가 지표를 최적화하는 랭킹 함수를 학습하는 지도 학습(Supervised Learning) 문제입니다.

## 3. LTR의 주요 과제: 평가 지표 최적화의 어려움

- **IR 평가 지표의 특성**:
    - AP (Average Precision), DCG 등 대부분의 IR 평가 지표는 문서의 **순서(Order)**에 의존합니다.
    - 랭킹 함수 $f(X)$ (여기서 $X$는 특징 벡터)의 작은 변화가 문서 순서의 큰 변화를 야기할 수 있고, 이는 평가 지표 값을 불연속적으로 변화시킵니다.
    - 즉, IR 평가 지표는 랭킹 함수 $f(X)$에 대해 **미분 가능(Differentiable)하거나 연속적(Continuous)이지 않은 경우가 많습니다.**
    - 이는 경사 하강법(Gradient Descent)과 같은 일반적인 최적화 알고리즘을 직접 적용하기 어렵게 만듭니다.

## 4. LTR의 주요 접근 방식

이러한 최적화의 어려움을 해결하기 위해, LTR은 목적 함수를 근사(Approximate)하는 다양한 방식을 사용합니다.

### 4.1. Pointwise LTR (점별 접근 방식)

- **핵심 아이디어**: 각 문서의 절대적인 관련성 점수(Relevance Score) 또는 관련성 등급(Relevance Label)을 정확하게 예측하는 것을 목표로 합니다. 이상적으로, 관련성 점수를 완벽하게 예측하면 완벽한 랭킹을 얻을 수 있습니다. ($f \rightarrow \text{score} \rightarrow \text{order} \rightarrow \text{metric}$)
- **랭킹 문제를 다음 문제로 환원**:
    - **회귀 (Regression)**: 문서의 관련성 등급(예: 0, 1, 2)을 실수 값으로 예측합니다. (논문: "Subset Ranking using Regression", D.Cossock and T.Zhang, COLT 2006)
        - 손실 함수 예: $\sum_i w_i |f(x_i) - y_i|$ (가중치 $w_i$를 두어 관련 문서에 더 큰 중요도 부여 가능)
    - **분류 (Classification) / 순서형 분류 (Ordinal Classification)**: 문서를 미리 정의된 관련성 등급(클래스)으로 분류합니다. (논문: "Ranking with Large Margin Principles", A. Shashua and A. Levin, NIPS 2002)
        - 목표: 문서를 올바른 카테고리에 배치하고 마진을 최대화.
- **장점**: 기존의 회귀/분류 알고리즘을 비교적 쉽게 적용할 수 있습니다.
- **단점**:
    - **IR 평가 지표 직접 최적화 불가**: 관련성 점수의 절대적인 값 예측에 초점을 맞추므로, 문서 간 상대적인 순서나 전체 목록의 품질을 직접적으로 고려하지 못합니다. (예: (0 → 1, 2 → 0) 예측 오류가 (0 → -2, 2 → 4) 예측 오류보다 점수 차이는 작지만, 실제 랭킹 품질에는 더 나쁠 수 있음)
    - **문서 위치 무시**: 높은 순위 문서의 오류와 낮은 순위 문서의 오류를 동일하게 취급하는 경향이 있습니다.
    - **쿼리별 문서 수 편향**: 문서 수가 많은 쿼리의 학습에 더 큰 영향을 받을 수 있습니다.

### 4.2. Pairwise LTR (쌍별 접근 방식)

- **핵심 아이디어**: 두 문서 쌍의 상대적인 순서(선호도)를 정확하게 예측하는 것을 목표로 합니다. 이상적으로, 모든 문서 쌍의 올바른 부분 순서(Partial Order)를 예측하면 완벽한 랭킹을 얻을 수 있습니다. ($f \rightarrow \text{partial order} \rightarrow \text{order} \rightarrow \text{metric}$)
- **순서형 회귀 (Ordinal Regression) 또는 선호도 학습 (Preference Learning)**:
    - 문서 쌍 $(d_i, d_j)$에 대해, $d_i$가 $d_j$보다 더 관련성이 높은지($d_i \succ d_j$)를 학습합니다.
    - 손실 함수 예: $\sum_{d_i \succ d_j} \mathbb{I}(f(x_i) < f(x_j))$ (잘못 정렬된 쌍의 개수 최소화 - 0/1 loss) 또는 이를 근사한 부드러운 손실 함수 사용.
- **주요 알고리즘**:
    - **RankSVM (Joachims, KDD’02)**:
        - 선호되는 쌍 $(d_i \succ d_j)$에 대해 $f(x_i) > f(x_j)$가 되도록 학습합니다 (마진 최대화).
        - 잘못 정렬된 쌍의 수를 최소화하는 것을 목표로 하며, Hinge Loss와 유사한 손실 함수를 사용합니다.
        - 클릭 데이터(Clickthrough Data)를 활용하여 상대적 선호도를 학습할 수 있습니다. (클릭된 문서 > 클릭되지 않은 문서)
        - 선형적인 특징 간의 상관관계를 학습합니다.
    - **RankBoost (Freund, Iyer, et al. JMLR 2003)**:
        - 잘못 정렬된 쌍에 대한 손실(Exponential Loss 등)을 부스팅(Boosting) 알고리즘을 통해 최적화합니다.
        - 여러 약한 랭커(Weak Ranker, 예: 단일 특징 기반)를 결합하여 강한 랭커를 만듭니다.
    - **RankNet (Burges et al., ICML 2005)**: 신경망을 사용하여 쌍별 확률(두 문서 중 하나가 더 관련성이 높을 확률)을 모델링하고, Cross-Entropy 손실을 최소화합니다.
    - **GBDT (Gradient Boosting Decision Tree) 기반 Pairwise LTR (Zheng et al. SIRIG’07)**:
        - Pairwise Hinge Loss와 유사한 목적 함수를 GBDT를 사용하여 최적화합니다.
        - 특징들의 비선형적인 조합을 학습할 수 있습니다.
- **장점**:
    - Pointwise 방식보다 랭킹의 본질(상대적 순서)에 더 가깝습니다.
    - 클릭 데이터 등 암묵적 피드백(Implicit Feedback)을 활용하기 용이합니다.
- **단점**:
    - 여전히 전체 랭킹 목록의 품질(예: NDCG)을 직접 최적화하지는 못합니다.
    - 문서의 절대적인 위치 정보는 여전히 간과될 수 있습니다. (예: 최상위 결과에서의 순서 오류와 하위 결과에서의 순서 오류를 동일하게 취급)

### 4.3. Listwise LTR (목록별 접근 방식)

- **핵심 아이디어**: 개별 문서나 문서 쌍이 아닌, **전체 랭킹 목록(List of Documents)**을 직접적으로 고려하여 IR 평가 지표를 최적화하는 것을 목표로 합니다. ($f \rightarrow \text{order} \rightarrow \text{metric}$)
- **주요 과제**: IR 평가 지표의 불연속성/미분 불가능성 문제를 해결해야 합니다.
- **접근 방식**:
    - **평가 지표를 근사하거나 부드럽게(Smooth) 만들기**:
        - **SoftRank (Taylor et al., WSDM'08)**: 미분 불가능한 순위 기반 지표를 부드럽게 근사하여 최적화.
    - **순열(Permutation)에 대한 손실 함수 정의**:
        - **ListNet (Cao et al., ICML'07)**: 랭킹 목록의 확률 분포를 정의하고, 실제 랭킹 목록과의 KL-Divergence 등을 최소화.
    - **기존 손실 함수에 평가 지표 정보를 주입 (Gradient Injection)**:
        - **LambdaRank (Burges et al., NIPS 2006)**:
            - RankNet의 쌍별 Cross-Entropy 손실 함수의 그래디언트에, 특정 문서 쌍 $(d_i, d_j)$의 순서가 바뀌었을 때 IR 평가 지표(예: NDCG)의 변화량($\Delta \text{NDCG}$)을 곱하여 사용합니다.
            - 즉, 평가 지표에 더 큰 영향을 미치는 쌍의 순서 오류에 더 큰 가중치를 부여하여 학습합니다.
            - 목적 함수는 명시적으로 정의되지 않지만, 그래디언트를 통해 간접적으로 IR 지표를 최적화합니다.
        - **LambdaMART (Burges, 2010)**: LambdaRank의 그래디언트를 MART(Multiple Additive Regression Trees, GBDT의 일종)와 결합한 알고리즘. 현재 LTR 분야에서 가장 강력한 성능을 보이는 알고리즘 중 하나입니다.
    - **구조적 SVM (Structural SVM) 활용**:
        - **SVM-MAP (Yue et al., SIGIR’07)**: MAP(Mean Average Precision)를 직접 최적화하기 위해 구조적 SVM을 사용합니다.
            - 목표: 실제 정답 랭킹과 다른 모든 (잘못된) 랭킹 사이의 마진을 최대화.
            - 가장 제약을 위반할 가능성이 높은 랭킹(Most Violated Constraint)을 효율적으로 찾아 학습에 활용.
- **장점**:
    - 문서의 절대적인 위치와 전체 목록의 품질을 고려하여 IR 평가 지표를 직접적으로 최적화하려는 시도입니다.
    - 가장 이론적으로 정교하고 실제 성능도 우수한 경우가 많습니다.
- **단점**:
    - 최적화해야 하는 공간(예: 모든 가능한 순열)이 매우 커서 계산적으로 복잡할 수 있습니다.
    - 알고리즘 설계가 Pointwise나 Pairwise보다 더 어렵습니다.

## 5. LTR의 실제 적용 및 전망

- **실험적 비교**: 일반적으로 Listwise > Pairwise > Pointwise 순으로 좋은 성능을 보이는 경향이 있지만, 데이터셋이나 특징에 따라 달라질 수 있습니다. LambdaMART, RankBoost 등이 우수한 성능
- **전통적 IR과의 연결**: LTR은 확률적 랭킹 원리(Probabilistic Ranking Principle - PRP)와도 연결되며, 문서와 쿼리 간의 관련성을 $P(R=1|Q,D)$로 모델링하는 조건부 모델로 볼 수 있습니다.
- **광범위한 특징 활용**: 텍스트 매칭 점수뿐만 아니라, 링크 구조(PageRank), 쿼리 관계, 사용자 클릭/조회 기록, 시각 정보, 소셜 네트워크 정보 등 다양한 정보를 특징으로 활용하여 랭킹 품질을 향상시킬 수 있습니다.
- **향후 연구 방향**: 더 정확한 손실 함수 근사, 더 빠른 최적화 알고리즘, 대규모 데이터 처리, 다양한 응용 시나리오 확장 등.
- **LTR 관련 자원**:
    - **서적**: Liu, Tie-Yan. "Learning to rank for information retrieval." (2011); Li, Hang. "Learning to rank for information retrieval and natural language processing." (2011).
    - **패키지**: RankLib, SVM^light (RankingSVM).
    - **데이터셋**: LETOR, Yahoo! Learning to Rank Challenge.

## 6. 관련성 피드백의 필요성 및 기본 개념

- **IR 시스템은 상호작용 시스템**: 사용자의 정보 요구(Information Need)와 시스템이 추론한 정보 요구 사이에는 항상 **간극(GAP)**이 존재할 수 있습니다. 관련성 피드백은 이 간극을 줄이기 위한 중요한 메커니즘입니다.
- **관련성 피드백 (Relevance Feedback)이란?**:
    - 사용자가 초기 검색 결과에 대해 관련 있음(Relevant, +) 또는 관련 없음(Non-relevant, -)과 같은 **판단(Judgment)**을 제공하면, 시스템은 이 정보를 활용하여 **쿼리를 수정하거나 랭킹 모델을 업데이트**하여 더 나은 검색 결과를 제공하는 과정입니다.
- **피드백의 기본 아이디어**:
    1. **쿼리 확장 (Query Expansion)**:
        - 사용자가 관련 있다고 판단한 문서들에서 중요한 단어(관련 검색어)를 추출하여 원래 쿼리에 추가합니다.
        - 예: 쿼리="information retrieval" → 관련 문서에서 "search", "search engine", "ranking", "query" 등의 단어 발견 → 확장된 쿼리 생성.
        - 목표: 재현율(Recall)을 높이고, 때로는 정밀도(Precision)도 향상시킬 수 있습니다.
    2. **학습 기반 검색 (Learning-based Retrieval)**:
        - 사용자의 피드백 문서를 랭킹 모델 업데이트를 위한 지도 신호(Supervision)로 사용합니다. (이전 "Learning to Rank" 강의에서 다룬 내용)
- **실제 시스템에서의 관련성 피드백**:
    - 과거 Google 검색 결과 옆에 "유사한 페이지", "이 페이지 제외"와 같은 기능이 있었습니다. (현재는 다른 형태로 발전)
    - 이미지 검색 시스템에서 "비슷한 이미지 찾기" 기능 등으로 널리 사용됩니다.

## 7. 의사 관련성 피드백 (Pseudo Relevance Feedback / Blind Feedback)

- **문제점**: 사용자가 명시적인 피드백을 제공하는 것을 꺼리거나 번거로워할 수 있습니다.
- **의사 관련성 피드백**:
    - 사용자의 명시적인 피드백 없이, 초기 검색 결과 중 **상위 k개의 문서를 관련 있다고 가정**하고, 이 문서들을 사용하여 자동으로 쿼리를 확장하거나 모델을 개선합니다.
    - 초기 검색 결과의 품질이 어느 정도 보장될 때 효과적일 수 있습니다.

## 8. 피드백 기술: 쿼리 확장 중심

- **피드백을 통한 쿼리 확장 절차**:
    1. **용어 선택 (Term Selection)**: 피드백 문서(관련 있다고 판단된 문서 또는 상위 k개 문서)에서 쿼리 확장에 사용할 중요한 용어를 선택합니다.
    2. **쿼리 확장 (Query Expansion)**: 선택된 용어를 원래 쿼리에 추가합니다.
    3. **쿼리 용어 가중치 재조정 (Query Term Re-weighting)**: 확장된 쿼리의 각 용어에 새로운 가중치를 부여합니다.
- **피드백을 학습 신호로 활용**: Learning to Rank에서 다룸.

## 9. 벡터 공간 모델 (VSM)에서의 관련성 피드백: 로키오 알고리즘 (Rocchio Algorithm)

- **일반적인 아이디어**: 쿼리 벡터를 수정합니다.
    - 새로운 (가중치가 적용된) 용어를 추가합니다.
    - 기존 용어의 가중치를 조정합니다.
- **로키오 알고리즘 (Rocchio, 1971)**: 가장 잘 알려지고 효과적인 VSM 피드백 방법.
    - **원리**: 수정된 쿼리 벡터($\vec{q}_m$)는 원래 쿼리 벡터($\vec{q}_o$)를 관련 문서들의 중심(Centroid) 방향으로 이동시키고, 비관련 문서들의 중심 방향에서 멀어지도록 조정합니다.
    - **로키오 공식**:
    $\vec{q}_m = \alpha \vec{q}*o + \beta \frac{1}{|D_r|} \sum*{\vec{d}*j \in D_r} \vec{d}j - \gamma \frac{1}{|D{nr}|} \sum*{\vec{d}*k \in D*{nr}} \vec{d}_k$
        - $D_r$: 관련 문서 집합, $D_{nr}$: 비관련 문서 집합.
        - $\alpha, \beta, \gamma$: 각 요소의 중요도를 조절하는 파라미터.
    - **실제 적용 시 고려사항**:
        - **비관련 문서의 영향력 감소**: 일반적으로 $\gamma$ 값을 작게 설정하거나 비관련 문서 항을 무시하는 경우가 많습니다. (이유: 비관련 문서는 매우 다양하여 중심을 잡기 어렵고, 잘못된 방향으로 쿼리를 이동시킬 수 있음)
        - **효율성**: 피드백 문서의 중심 벡터 계산 시, 가중치가 높은 소수의 단어만 고려하여 차원을 축소할 수 있습니다.
        - **훈련 편향 방지**: 원래 쿼리의 가중치($\alpha$)를 상대적으로 높게 유지하여 쿼리가 너무 많이 변형되는 것을 방지합니다.
    - 명시적 관련성 피드백과 의사 관련성 피드백 모두에 사용 가능하며, 일반적으로 안정적이고 효과적입니다.

## 10. 확률 모델에서의 관련성 피드백

- **고전적 확률 모델 (Robertson-Sparck Jones Model, 1976)**:
    - 문서와 쿼리가 주어졌을 때 문서가 관련 있을 확률 $P(R=1|Q,D)$과 관련 없을 확률 $P(R=0|Q,D)$의 비(Odds)를 사용하여 랭킹합니다.
    - 각 용어 $A_i$에 대해 두 개의 파라미터 추정:
        - $p_i = P(A_i=1|Q,R=1)$: 관련 문서에서 용어 $A_i$가 나타날 확률.
        - $u_i = P(A_i=1|Q,R=0)$: 비관련 문서에서 용어 $A_i$가 나타날 확률.
    - **RSJ 가중치 (랭킹 공식)**: $\sum_{i \in q \cap d} \log \frac{p_i(1-u_i)}{u_i(1-p_i)}$
    - **파라미터 추정 (피드백 활용)**: 사용자의 관련성 판단 정보를 이용하여 $p_i$와 $u_i$를 각 쿼리에 대해 추정합니다. (베이즈 추정 사전확률로 +0.5, +1 사용)
        - $\hat{p}_i = \frac{\text{#(rel doc with } A_i) + 0.5}{\text{#(rel doc)} + 1}$
        - $\hat{u}_i = \frac{\text{#(nonrel doc with } A_i) + 0.5}{\text{#(nonrel doc)} + 1}$
    - 피드백을 통해 현재 쿼리에 대한 $P(D|Q,R=1)$ (문서 생성 모델) 개선 가능.

## 11. 언어 모델 (Language Model)에서의 관련성 피드백

- **언어 모델 기반 검색**: 쿼리가 주어졌을 때 해당 쿼리를 생성할 확률이 높은 문서($P(Q|D)$)를 랭킹합니다. (Query Likelihood Model)
- **피드백 접근 방식**:
    - 확률적인 **쿼리 모델($\theta_Q$)**을 도입하고, 이 쿼리 모델과 문서 모델($\theta_D$) 간의 거리를 측정하여 랭킹합니다.
    - 피드백은 **쿼리 모델을 업데이트**하는 데 사용됩니다.
- **Kullback-Leibler (KL) 발산 기반 검색 모델**:
    - 문서 모델 $\theta_D$에 대한 쿼리 모델 $\theta_Q$의 KL 발산 값에 음수를 취한 것을 관련성 점수로 사용합니다. ($Sim(Q,D) \propto -D_{KL}(\theta_Q || \theta_D)$)
        - KL 발산: 두 확률 분포 간의 차이를 측정하는 비대칭적 척도. $D_{KL}(P||Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$.
    - 랭킹 공식 (근사): $Sim(Q,D) \propto \sum_{w \in Q \cap D} P(w|\theta_Q) \log P(w|\theta_D) - \sum_{w \in Q} P(w|\theta_Q) \log P(w|\theta_Q)$ (두 번째 항은 쿼리별로 동일하여 랭킹에 영향 없음)
        - $P(w|\theta_Q)$는 쿼리 단어의 경험적 분포(MLE)로 추정.
- **피드백을 통한 모델 보간 (Model Interpolation)**:
    - 새로운 쿼리 모델($\theta_Q'$)은 원래 쿼리 모델($\theta_Q$)과 피드백 문서로부터 생성된 피드백 모델($\theta_{Q_F}$)을 선형 보간하여 만듭니다.
    - $\theta_Q' = (1-\lambda) \theta_Q + \lambda \theta_{Q_F}$
        - $\lambda=0$: 피드백 없음.
        - $\lambda=1$: 완전 피드백 (피드백 모델만 사용).
    - 로키오 피드백과 매우 유사한 형태지만, 확률적 관점에서의 해석이 다릅니다.
- **피드백 모델($\theta_{Q_F}$) 추정 - 생성적 혼합 모델 (Generative Mixture Model) 및 EM 알고리즘**:
    - 가정: 피드백 문서는 "주제 단어(Topic Words)"와 "배경 단어(Background Words)"의 혼합으로 생성됩니다.
        - $P(w|Q_F) = (1-l)P(w|C) + l P(w|\text{query topic})$
            - $P(w|C)$: 배경 단어 분포 (전체 컬렉션의 단어 분포, 알려져 있음).
            - $P(w|\text{query topic})$: 주제 단어 분포 (추정해야 할 파라미터).
            - $l$: 피드백 문서 내 주제 단어의 비율 (Noise ratio의 반대).
    - 각 단어가 배경 단어인지 주제 단어인지 알 수 없으므로 (숨겨진 변수, Hidden Variable $z_i$), **EM (Expectation-Maximization) 알고리즘**을 사용하여 $P(w|\text{query topic})$와 $l$을 추정합니다.
        - **E-step (Expectation)**: 현재 파라미터 값을 기준으로 각 단어가 주제 단어일 확률 $P(z_i=0|w_i)$ (또는 배경 단어일 확률 $P(z_i=1|w_i)$)을 계산합니다.
            - $P(z_i=0|w_i) = \frac{l \cdot P(w_i|\text{query topic})}{ (1-l)P(w_i|C) + l \cdot P(w_i|\text{query topic}) }$
        - **M-step (Maximization)**: E-step에서 계산된 확률을 이용하여 $P(w|\text{query topic})$와 $l$을 재추정(MLE)합니다.
            - $P(w|\text{query topic})*{\text{new}} = \frac{\sum_j c(w_j, F) \cdot P(z_j=0|w_j)}{\sum*{w'} \sum_j c(w_j', F) \cdot P(z_j=0|w_j')}$
        - 수렴할 때까지 E-step과 M-step을 반복합니다.
    - **로키오와의 차이점**: EM 기반 방법은 각 단어가 주제 관련 단어인지 배경 단어인지의 "정체성(Identity)"을 구분하려 시도하지만, 로키오는 이러한 구분을 명시적으로 하지 않습니다.
- **부정적 피드백 처리**: 언어 모델 기반 피드백에서 부정적 피드백을 어떻게 효과적으로 통합할지는 여전히 연구가 필요한 영역입니다.

## 12. 요약 및 결론

정보 검색에서 Learning to Rank (LTR)는 다양한 특징들을 자동으로 결합하여 검색 성능을 최적화하는 핵심 기술입니다. Pointwise, Pairwise, Listwise 접근 방식 중 일반적으로 Listwise 방법이 가장 좋은 성능을 보이며, LambdaMART와 같은 알고리즘이 실제 검색 시스템에서 널리 사용됩니다.

또한 관련성 피드백 기술은 사용자의 정보 요구와 시스템이 추론한 정보 요구 사이의 간극을 줄이는 중요한 메커니즘입니다. 로키오 알고리즘과 같은 벡터 공간 모델 기반 피드백과 언어 모델 기반 피드백은 각각 쿼리 확장과 모델 개선에 효과적이며, 의사 관련성 피드백은 명시적인 사용자 입력 없이도 검색 성능을 향상시킬 수 있는 실용적인 방법입니다.
