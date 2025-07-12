# concrete-crack-detection
[2024-2] 서울여자대학교 딥러닝기반데이터분석 과제2

딥러닝 모델을 활용하여 콘크리트 구조물의 표면 균열 여부를 탐지하는 프로젝트입니다.
이 프로젝트는 토목 구조물의 주요 결함인 콘크리트 표면의 균열을 감지하여 건물의 상태를 평가하고 구조적 안전성 확보를 목표로 합니다.

---

## 프로젝트 개요

- **과제명**: 딥러닝 모델을 활용한 콘크리트 표면 균열 감지
- **수행자**: 데이터사이언스학과 박서진
- **주요 기술 스택**:
  - Python
  - Pytorch
  - FastAPI
- **활용 모델**:
  - MobileNetV2 (시도만 했고 최종 모델로 사용하지 않음)
  - ResNet18 (최종 선택 모델)
 
---

## 데이터셋

### Surface Crack Detection Dataset

- 출처: [Surface Crack Detection Dataset on Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
- 총 이미지 수: 40,000장
  - Positive (균열 있음): 20,000장
  - Negative (균열 없음): 20,000장
- 이미지 크기: 227 x 227
- 다양한 표면 마감과 조명 조건에서 촬영된 고해상도 이미지

### crack_1000 샘플 데이터

- Kaggle 원본 데이터셋에서 클래스별로 1,000장씩 샘플링하여 사용
- 학습시간을 줄이고 실험 편의성을 높이기 위함
- 폴더 구조:
  '''
  crack_1000/
        ├── positive_1000/   # 균열 있음 이미지 1,000장
        └── negative_1000/   # 균열 없음 이미지 1,000장
  '''
- 본 깃허브에는 이미지 파일이 포함되어 있지 않습니다. 위의 링크의 Kaggle에서 직접 다운로드하여 사용하세요.

---

## 실험 내용 및 결과

### MobileNetV2 실험

- crack_1000 데이터셋으로 학습
- 초기 에폭에서 빠르게 손실 감소, Validation Accuracy가 99% 이상 도달
- 그러나 Validation Loss에서 불안정하고 과적합 발생
- 최종적으로 ResNet18이 더 안정적 성능을 보여 MobileNetV2는 사용하지 않음

#### 학습 그래프 예시




