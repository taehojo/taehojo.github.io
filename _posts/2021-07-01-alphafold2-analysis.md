---
layout: post
title: "알파폴드2 모델 분석"
date: 2021-07-01
description: "2021년 7월 15일, 네이처지에 발표된 알파폴드2 논문의 내용 중, 알파폴드2가 작동하는 흐름에 초점을 두고 요약되었습니다."
---

> 2021년 7월 15일, 네이처지에 발표된 알파폴드2 논문의 내용 중, 알파폴드2가 작동하는 흐름에 초점을 두고 요약되었습니다.

## 알파폴드2의 작동 원리

![Figure 1](/alphafold/images/image-a1.png)

**Figure 1**. 알파폴드2의 model architecture

이 그림은 알파폴드2의 처리 과정을 한눈에 보여줍니다. 알파폴드2는 다음의 3가지 스텝으로 구성되어 있습니다. 먼저 입력 데이터를 전처리하는 Input feature embeddings(①) 단계, 어텐션 학습을 통해 전처리된 데이터에서 필요한 정보를 뽑아내는 Evoformer(②)단계, 그리고 여기서 나온 정보를 구체적인 3차원 좌표로 처리하는 Structure module(③)단계입니다.

## 1. Input feature embeddings

![Figure 2](/alphafold/images/image-a2.png)

**Figure 2**. Input feature embeddings

알파폴드2의 첫 단계는 알파폴드1과 마찬가지로 입력 시퀀스를 전처리 하는 단계입니다. 먼저 유전자 데이터베이스에서 유사한 evolutionary 서열을 검색하여 다중 서열 정렬(MSA)을 생성합니다(①). 170,000개의 PDB 데이터, Uniprot의 대규모 데이터베이스를 이용했으며, JackHMMER와 HHblits를 이용해서 UniRef90, MGnify clusters, BFD를 검색하는 방식을 썼습니다.

알파폴드2는 여기에 추가로 쿼리 시퀀스와 유사한 시퀀스 부분을 가진, 알려진 단백질 템플릿을 검색합니다(②). 알파폴드2가 기존의 Yang Zhang 랩이나 David Baker랩을 뛰어 넘을 수 있었던 건 시퀀스 정보에서 필요한 정보를 추가하는 Extra MSA Stack(③), Evoformer Stack(④)과정이 있다는 것입니다.

## 2. Evoformer

Pair representation과 MSA representation이 만들어지면 이 정보는 Evoformer단계로 넘어갑니다.

![Figure 3](/alphafold/images/image-a3.png)

**Figure 3**. Evoformer block

주어진 Pair representation과 MSA representation을 개선하기 위해 어텐션 메커니즘의 48개 레이어로 구성된 deep tranformer-like 네트워크(①)가 적용되는 단계입니다. 48개의 모든 레이어에는 각각 개별 매개변수가 있으며 입력과 출력은 MSA representation과 Pair representation입니다.

Evoformer 안에는 두개의 흐름이 있는데, 하나는 MSA representation이 입력되어 진행되는 흐름으로 위쪽의 흐름입니다(②). 또 하나는 Pair representation(③)의 정보 흐름입니다. 이 두 흐름의 아이디어는 기본적으로 단백질의 공간적, 진화적 관계에 대한 직접적인 추론을 가능하게 하는 정보를 교환하게 하는 것입니다.

## 3. Structure module

Evoformer단계에서 만들어진 정보는 protein geometry의 구체적인 3차원 좌표로 변환하는 Structure module단계로 넘어옵니다.

![Figure 4](/alphafold/images/image-a4.png)

**Figure 4**. Structure Module

Evoformer가 가지고 있는 정보는 2D representation 형태로, 반드시 3차원 단백질 기하학으로 변환되어야 합니다. 이는 weight을 공유하는 8개의 RNN 블록에서 수행됩니다. Evoformer의 최종 MSA representation 정보와 Pair representation 정보가 사용되며, distances, torsions, atom coordinates, Cα-lDDT의 추정치를 예측하게 됩니다.

Invariant Point Attention 메커니즘에는 단백질 구조 개선을 위해 특별히 설계된 여러가지 기술이 포함되어 있습니다. 이중 하나가 시퀀스 표현과 백본 변환을 반복적으로 업데이트함으로써 단백질의 블랙홀 초기화가 최종 백본 기하학이 나타날 때까지 점진적으로 조정되게 하는 것입니다.

## 마치며

알파폴드는 CASP14에서 0부터 100까지의 범위를 가지는 GDT점수(100이 만점)에서 median score 92.4점을 얻었습니다. 이는 약 1.6 Angstroms의 평균 오차(RMSD)를 가진다는 의미인데, 곧 원자의 너비 (0.1 나노 미터)와 비슷한 수준입니다.

결론적으로 Attention-based neural network을 통해 End-to-end 방식으로 학습했다고 할 수 있으나, 이러한 간단한 설명으로는 턱없이 부족할 만큼 획기적인 아이디어들이 적재 적소에 사용된 혁신적인 시스템이라는 생각입니다.

1972년 아미노산 배열의 1차 구조가 처음으로 해석된 이래, 단백질의 구조를 알아내기 위한 노력이 계속되어 왔습니다. 알파폴드2는 지난 50여년간 이어온 분자 생물학, 유전학, 생명정보학의 성과가 AI의 급속한 발전과 더불어 탄생한 놀라운 사건이 아닐 수 없습니다.
