---
layout: default
title: 딥러닝 + 알츠하이머 원인 단백질 추적
parent: 딥러닝 생명과학
nav_order: 2
---

# 딥러닝 + 알츠하이머 원인 단백질 추적

<br/>

> **질문**: 딥러닝으로 알츠하이머 병의 원인 단백질이 축적되는 형태와 위치를 파악할 수 있을까? <br/>
>
> **논문의 결론**:  '설명가능한 AI' 방법 중 하나인 LRP를 써서 타우 단백질이 알츠하이머로 발병되는 과정에서 축적되는 위치를 파악할 수 있으며, 이 정보를 가지고 알츠하이머 병 환자를 90.8%의 정확도로 예측할 수 있었습니다.  

<br/>

## **딥러닝을 이용한 알츠하이머 원인 단백질 추적**

알츠하이머병(AD)은 치매의 가장 흔한 유형으로, 일반적으로 기억 상실에 이어지는 점진적 인지 저하 와 기능 장애를 특징으로 합니다. 알츠하이머병에 대한 잠재적 치료법에 대한 많은 임상 시험이 있었으나 아직까지 완전한 치료법은 없습니다. 알츠하이머 병을 조기 발견 하고 병의 발병을 이해하기 위한  바이오마커를 찾는 것은 그래서 약물 개발 및 임상 시험에 있어 매우 중요합니다. 그동안 대부분의 바이오마커 연구는 베타 아밀로이드에 초점을 두었습니다. 이 연구는 딥러닝 프레임워크를 이용해 타우 단백질이 축적되는 유형과 알츠하이머병의 발병의 관계를 파악한  "Deep learning detection of informative features in tau PET for Alzheimer’s disease classification"(2020) 논문을 옮긴 것입니다. 

> <details>
> <summary>논문 인용 하기 </summary>
> <br>
> Jo, Taeho, et al. "Deep learning detection of informative features in tau PET for Alzheimer’s disease classification." BMC bioinformatics 21.21 (2020): 1-13.
> </details>

이 연구는 타우 양전자 방출 단층 촬영(PET) 이미지를 사용하여 알츠하이머병을 분류하고 XAI기법을 이용해 필요한 정보를 획득하는 딥러닝 기반 프레임워크를 소개하고 있습니다. 

먼저 인지적 정상(CN)인 참가자의 이미지와 알츠하이머병의 진단을 받은 참가자의 이미지를 예측하기 위해서 3D CNN(컨볼루션 신경망) 기반 분류 모델을 사용했고, 교차 검증을 기반으로 90.8%의 평균 정확도를 얻었습니다. 

이어서 딥러닝 분류 모델의 판정 기여도를 설명하는 LRP 모델을 이용해 알츠하이머병 환자의 이미지를 예측하는데 가장 크게 기여한 타우 PET 이미지의 뇌 영역을 시각화했습니다. 확인된 중요 영역에는  hippocampus, para hippocampus, thalamus 그리고 fusiform이 포함 되어 있었습니다. 

LRP 결과는 SPM12의 복셀별 분석 결과와 상당부분 일치했으며, 특히 entorhinal cortex 를 포함한 bilateral temporal lobes에서 알츠하이머병 관련 국소 타우 침착을 보여주었습니다. 딥러닝 분류 모델에 의해 계산된 알츠하이머병 확률 점수는 MCI 참가자의 medial temporal lobe에서 뇌 타우 침착과 상관관계가 있었습니다.

결론적으로 이 연구는 3D CNN 및 LRP 알고리즘을 결합한 딥러닝 프레임워크를 타우 PET 이미지에 적용하여 알츠하이머 병의 분류를 위한 유익한 정보를 식별할 수 있었음을 보여주며 이는 알츠하이머병의 조기 감지에 적용할 수 있음을 보여주고 있습니다.

<br/>**Deep learning detection of informative features in tau PET for Alzheimer’s disease classification** <br/>
Taeho Jo, Kwangsik Nho, Shannon L. Risacher and Andrew J. Saykin
<br/>

---

## Abstract

**Background**

Alzheimer’s disease (AD) is the most common type of dementia, typically characterized by memory loss followed by progressive cognitive decline and functional impairment. Many clinical trials of potential therapies for AD have failed, and there is currently no approved disease-modifying treatment. Biomarkers for early detection and mechanistic understanding of disease course are critical for drug development and clinical trials. Amyloid has been the focus of most biomarker research. Here, we developed a deep learning-based framework to identify informative features for AD classification using tau positron emission tomography (PET) scans.

**Results**

The 3D convolutional neural network (CNN)-based classification model of AD from cognitively normal (CN) yielded an average accuracy of 90.8% based on five-fold cross-validation. The LRP model identified the brain regions in tau PET images that contributed most to the AD classification from CN. The top identified regions included the hippocampus, parahippocampus, thalamus, and fusiform. The layer-wise relevance propagation (LRP) results were consistent with those from the voxel-wise analysis in SPM12, showing significant focal AD associated regional tau deposition in the bilateral temporal lobes including the entorhinal cortex. The AD probability scores calculated by the classifier were correlated with brain tau deposition in the medial temporal lobe in MCI participants (r = 0.43 for early MCI and r = 0.49 for late MCI).

**Conclusion**

A deep learning framework combining 3D CNN and LRP algorithms can be used with tau PET images to identify informative features for AD classification and may have application for early detection during prodromal stages of AD.

## Background

The accumulation of hyperphosphorylated and pathologically misfolded tau protein is one of the cardinal and most common features in Alzheimer’s disease (AD) [[1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR1),[2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR2),[3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR3),[4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR4),[5](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR5)]. The amount and spatial distribution of abnormal tau, seen pathologically as neurofibrillary tangles in brain, is closely related to the onset of cognitive decline and the progression of AD. The identification of morphological phenotypes of tau on in vivo neuroimaging may help to differentiate mild cognitive impairment (MCI) and AD from cognitively normal older adults (CN) and provide insights regarding disease mechanisms and patterns of progression [[6](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR6),[7](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR7),[8](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR8),[9](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR9)].

Deep learning has been used in a variety of applications in response to the increasingly complex and growing amount of medical imaging data [[10](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR10),[11](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR11),[12](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR12)]. Significant efforts have been made regarding the application of deep learning to AD research, but predicting AD progression through deep learning using neuroimaging data has focused primarily on magnetic resonance imaging (MRI) and/or amyloid positron emission tomography (PET) [[10](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR10), [13](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR13)]. However, MRI scans cannot visualize molecular pathological hallmarks of AD, and amyloid PET cannot, without difficulty, visualize the progression of AD due to the accumulation of amyloid-β early in the disease course with a plateau in later stages [[14](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR14), [15](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR15)].

The presence and location of pathological tau deposition in the human brain are well established [[2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR2), [3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR3), [5](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR5)]. Braak and Braak [[5](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR5)] analyzed AD-related neuropathology and generated a staging algorithm to describe the tau anatomical distribution [[6](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR6), [8](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR8), [16](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR16), [17](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR17)]. Their results have been confirmed by subsequent studies showing that the topography of tau corresponds with the pathological stages of neurofibrillary tangle deposition. Cross-sectional autopsy data shows that AD-related tau pathology may begin with tau deposition in the medial temporal lobe (Braak stages I/II), then moves to the lateral temporal cortex and part of the medial parietal lobe (stage III/IV), and eventually to broader neocortical regions (V / VI).

In this study, we developed a novel deep learning-based framework that identifies the morphological phenotypes of tau deposition in tau PET images for the classification of AD from CN. Application of CNN to tau PET is novel as the spatial characteristics and interpretation are quite different compared to amyloid PET, fluorodeoxyglucose (FDG) PET, or MRI. In particular, the regional location and topography of tau PET signal is considered to be more important than for other molecular imaging modalities. This has implications for how CNN interacts with the complex inputs as well as for visualization of informative features. The deep learning-derived AD probability scores were then applied to prodromal stages of disease including early and late mild cognitive impairment (MCI).

## Methods

**Study participants**

All individuals included in the analysis were participants in the Alzheimer’s Disease Neuroimaging Initiative (ADNI) cohort [[18](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR18), [19](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR19)]. A total of 300 ADNI participants (N = 300; 66 CN, 66 AD, 97 early mild cognitive impairment (EMCI), and 71 late MCI (LMCI)) with [18F]flortaucipir PET scans were available for analysis [[1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR1)]. Genotyping data were also available for all participants [[19](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR19)]. Informed consent was obtained for all subjects, and the study was approved by the relevant institutional review board at each data acquisition site.

**Alzheimer’s Disease Neuroimaging Initiative (ADNI)**

ADNI is a multi-site longitudinal study investigating early detection of AD and tracking disease progression using biomarkers (MRI, PET, other biological markers, and clinical and neuropsychological assessment) [[1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR1)]. Demographic information, PET and MRI scan data, and clinical information are publicly available from the ADNI data repository (https://www.loni.usc.edu/ADNI/).

**Imaging processing**

Pre-processed [18F]flortaucipir PET scans (N = 300) were downloaded from the ADNI data repository, one scan per individual. Scans were normalized to Montreal Neurologic Institute (MNI) space using parameters generated from segmentation of the T1-weighted MRI scan in Statistical Parametric Mapping v12 (SPM12) ([www.fil.ion.ucl.ac.uk/spm/](http://www.fil.ion.ucl.ac.uk/spm/)). Standard uptake value ratio (SUVR) images were then created by intensity-normalization using a cerebellar crus reference region.

**Deep learning method for AD classification**

Deep learning is a subset of machine learning that has been applied in various fields [[20](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR20), [21](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR21)]. Deep learning uses a back-propagation procedure [[22](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR22)], which utilizes gradient descent for the efficient error functions and gradient computing [[10](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR10), [23](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR23),[24](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR24),[25](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR25),[26](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR26)]. The weights are updated after the initial error value is calculated by the least squares method until the differential value becomes 0, as in the following formula:

![Q1](./images/tau1.png)

![Q1](./images/tau2.png)

*Net* is a sum of weights and bias, and *Yoj* is an output of neuron *j*. Convolutional Neural Network (CNN) is a method of inserting convolution and pooling layers to the basic structure of this neural network to reduce complexity. Since CNN is widely used in the field of visual recognition, we used a CNN method for the classification of AD from CN [[27](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR27)]. The overall architecture of 3D CNN that we used is shown in Fig. [1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig1). To avoid excessive epochs that can lead to overfitting, an early stopping method was applied to cease training if the model did not show improvement over 10 iterations. The learning rate of 0.0001 and Adam, a first-order gradient based probabilistic optimization algorithm [[28](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR28)] with a batch size of 4, were used for training a model. Feature maps (8, 16, and 32 features) were extracted from three hidden layers, with Maxpool3D and BatchNorm3D applied to each layer [[29](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR29)]. Dropout (0.4) was applied to the second and third layers. Five-fold cross validation was applied to measure the classifier performance for distinguishing AD from CN. All participants were partitioned into 5 subsets randomly, but every subset has the same ratio of CN and AD participants. One subset was selected for testing and the remaining four subsets were used for training. Among the four subsets for training, one subset (validation) was used without applying augmentation for tuning the weights of the layers without overfitting and the remaining three subsets were augmented by three criteria: flipping the image data, shifting the position within two voxels, and shifting the position simultaneously with the flip. Each fold was repeated four times for a robustness check, and the mean accuracy of the four repeats was used as the final accuracy. Pytorch 1.0.1 was used to design neural networks and load pre-trained weights, and all of the programs were run on Python 3.5.



[![figure1](https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12859-020-03848-0/MediaObjects/12859_2020_3848_Fig1_HTML.png)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0/figures/1)

**Figure 1** 3D convolutional neural network (3D-CNN)-based and layer-wise relevance propagation (LRP)-based framework for the classification of Alzheimer’s disease and the identification of informative features

<br/>

**Application of the AD-CN derived classification model to MCI**

After an AD classification model was constructed using AD and CN groups, the model was applied to the tau PET scans from the MCI participants to calculate AD probability scores. The AD probability scores were distributed from 0 to 1, and individuals with AD probability scores closer to 1 were classified as having AD characteristics, and individuals with scores closer to 0 were classified as having CN characteristics.

**Identification of informative features for AD classification**

We applied a layer-wise relevance propagation (LRP) algorithm to identify informative features and visualize the classification results [[30](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR30), [31](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR31)]. The LRP algorithm is used to determine the contribution of a single pixel of an input image to the specific prediction in the image classification task (for full details of the LRP algorithm, see [[30](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR30)]).

The output *xj* of a neuron *j* is calculated by a nonlinear activation function *g* and function *h* such as

![Q1](./images/tau3.png)

 So, the input value of the neuron *j* can be expressed as the following equation:

![Q1](./images/tau4.png)

Bach, et al. [[30](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR30)] proposed the following formula:

![Q1](./images/tau5.png)

Here, z+ijzij+ represents the positive input that the node *i* contributes to the node *j*, and z−ijzij− represents the negative input. The variable β that ranges from 0 to 1 controls the inhibition of the relevance redistribution. A larger β value (e.g. β = 1) makes the heat map clearer [[31](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR31)]. In this experiment, we set β = 1.

**Whole-brain imaging analysis**

A voxel-wise whole brain analysis to identify brain regions in the tau PET SUVR images showing significantly higher tau deposition in AD relative to CN was conducted in SPM12. The analysis was masked for grey plus white matter. The voxel-wise family-wise error (FWE) correction was applied at p < 0.05, with a cluster size of ≥ 50 voxels for adjustment for multiple comparisons.

## Results

In the analysis, 300 ADNI participants (66 CN, 66 AD, 97 EMCI, and 71 LMCI) who had baseline tau PET scans were used. Sample demographics were given in Table [1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Tab1).

![Q1](./images/taut1.png)

**Table 1 ** Demographic information

<br/>

**Classification of AD from CN**

We developed an image classifier to distinguish AD from CN by training a 3D CNN-based deep learning model on tau PET images. As the number of individuals with AD who had tau PET data was smaller than those of CN, we chose the same number of CN randomly (66 CN) to train a classifier with a balanced dataset. In the binary classification problem, it is a well-known issue that detecting disease when the majority of the applicants are healthy, the majority group may be referred as cases, causing biased classification [[32](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR32)]. So we used a random under-sampling (RUS) method to decrease samples from the majority group. All analyses were performed using five-fold cross-validation to reduce the likelihood of overfitting. Ultimately, cross validation in a novel independent data set will be important when such data becomes available. The classification accuracy is shown in Table [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Tab2). Our deep learning-based classification model of AD from CN yielded an average accuracy of 90.8% and a standard deviation of 2% from five-fold cross-validation (Table [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Tab2)).

![Q1](./images/taut2.png)

**Table 2** All participants were partitioned into 5 subsets randomly, but every subset has the same ratio of CN and AD participants

<br/>

**Identification of informative features for AD classification**

The LRP algorithm generated relevance heatmaps in the tau PET image to identify which brain regions play a significant role in a deep learning-based AD classification model. After selecting an AD classification model with the highest accuracy in each fold, we generated five heatmaps and selected the top ten regions with the highest contribution. Figure [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig2)a shows a visualization of the relevance heatmap in three orientations of our 3D CNN-based classification of AD from CN. The heatmap displays the primary brain regions that contributed to the classification, color-coded with increasing values from red to yellow. The colored regions in the heatmap include the hippocampus, parahippocampal gyrus, thalamus, and fusiform gyrus (Fig. [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig2)a). For comparison with our 3D CNN-based LRP results, Fig. [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig2)b shows the results of whole brain voxel-wise analysis in SPM12 to identify brain regions where there are significant differences between AD and CN in brain tau deposition (FWE corrected *p* value < 0.05; minimum cluster size (k) = 50). AD had significantly higher tau deposition in widespread regions including the bilateral temporal lobes with global maximum differences in the right and left parahippocampal regions, compared to CN (Fig. [2](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig2)b). The informative regions for AD classification in the LRP results are very similar to those found using SPM12, but the 3D CNN-based LRP identified smaller focal regions.



[<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12859-020-03848-0/MediaObjects/12859_2020_3848_Fig2_HTML.png" alt="figure2" style="zoom:80%;" />](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0/figures/2)

**Figure 2** Heatmaps of 3D-CNN classifications compared to voxel-wise group difference maps between AD and CN participant groups. **a** Relevance heatmaps of 3D-CNN classification of AD and CN. The bright areas represent the regions that most contribute to the CN/AD classification in CNN. Selected regions with the highest contribution include the hippocampus, parahippocampal gyrus, thalamus, fusiform gyrus, and diencephalon. **b** SPM maps show similar regions of the brain as the 3D-CNN maps where tau deposition is significantly higher in the AD group compared to the CN group (Voxel-wise FWE-corrected *p* value < 0.05; minimum cluster size (k) = 50)

<br/>

**Classification of MCI based on the AD-CN classification model**

We calculated the AD probability scores of MCI participants (97 EMCI and 71 LMCI, separately) using the classification model generated above. Figure [3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig3)a and b show scatter plots between the AD probability scores of EMCI and LMCI, respectively, with bilateral mean tau deposition in the medial temporal lobe (includes the entorhinal cortex, fusiform, and parahippocampal gyri). The correlation coefficients were R = 0.43 for EMCI and R = 0.49 for LMCI, with greater tau deposition levels in the medial temporal lobe associated with higher AD probability scores. Figure [3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig3)c and d show mean tau accumulation in the medial temporal cortex of EMCI and LMCI, respectively, for participants with AD probability score ranges (0 ≤ AD probability score ≤ 0.05 versus 0.95 ≤ AD probability score ≤ 1.00, the ranges to which 65% of EMCI and 62% of LMCI belong; 0 ≤ AD probability score < 0.5 versus 0.5 < AD probability score ≤ 1.00). In EMCI (Fig. [3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig3)c), a comparison between participants with 0 ≤ AD probability score ≤ 0.05 and those with 0.95 ≤ AD probability score ≤ 1.00 yielded a difference of 0.19 SUVR in the medial temporal lobe. In LMCI (Fig. [3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig3)d), the comparison of participants with low AD probability scores (0 ≤ AD probability score ≤ 0.05) and LMCI with high AD probability scores (0.95 ≤ AD probability score ≤ 1.00) yielded a difference of 0.26 SUVR in the medial temporal cortex. Whole brain voxel-wise analysis in SPM12 was performed to identify brain regions showing differences between tau deposition between MCI participants with low AD probability scores (0 ≤ AD probability score ≤ 0.05) and those with high AD probability scores (0.95 ≤ AD probability score ≤ 1.00). In EMCI (Fig. [4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig4)a, c) and LMCI (Fig. [4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig4)b, d), voxel-wise analysis identified significant group differences in the bilateral temporal lobes including the entorhinal cortex. In addition, the differences in tau deposition were more widespread in LMCI compared to EMCI (Fig. [4](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#Fig4)).



[<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12859-020-03848-0/MediaObjects/12859_2020_3848_Fig3_HTML.png" alt="figure3" style="zoom: 80%;" />](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0/figures/3)

**Figure 3** Results of scoring all the images through the classifier and comparing the scores to the tau accumulation in the MTL region. Correlation of DL score with the amount of tau accumulated in the MTL region was R = 0.43 for EMCI (**a**) and R = 0.49 for LMCI (**b**). The red and blue bar chart in C and D show the average of the tau amounts of the image with the DL score of 5% and the image of the top 95%. The charts in light red and light blue in C and D are the result of averaging the bottom 50% and top 50% of the images

<br/>



[<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12859-020-03848-0/MediaObjects/12859_2020_3848_Fig4_HTML.png" alt="figure4" style="zoom: 80%;" />](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0/figures/4)

**Figure 4** Voxel-wise differences between MCI participants with AD-like tau patterns and CN-like tau patterns defined using the 3D-CNN classifier. Significantly greater tau was observed in EMCI (**a**, **c**) and LMCI (**b**, **d**) with high AD probability (“AD-like,” 0.95 ≤ AD probability score ≤ 1.00) relative to the low AD probability group (“CN-like,” 0 ≤ AD probability score ≤ 0.05). Voxel-wise significance maps are displayed at FWE corrected *p* value < 0.05; minimum cluster size (k) = 50

<br/>

## Discussion

We developed a deep learning framework for detecting informative features in tau PET for the classification of Alzheimer’s disease. After training a 3D CNN-based AD/CN classifier on 132 [18F]flortaucipir PET images to distinguish AD with > 90% accuracy, heatmaps were generated by a LRP algorithm to show the most important regions for the classification. This model was then applied to [18F]flortaucipir PET images from 168 MCI to classify them into “AD similar” and “CN similar” groups for further investigation of the morphological characteristics of the tau deposition.

Maass, et al. [[7](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR7)] examined the key regions of in vivo tau pathology in ADNI using a data-driven approach and determined that the major regions contributing to a high global tau signal mainly overlapped with Braak stage III ROIs (i.e., amygdala, parahippocampal gyri and fusiform). Our deep learning-based results correspond well to the pattern reported by Maass, et al. [[7](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR7)] on a more limited data set.

It is noteworthy that stages III / IV can be seen in both CN and AD patients, while stages I / II are common in CN and stages V / VI are common for AD patients [[3](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR3)]. Thus, it is difficult to predict AD by measuring tau deposition in stage III / IV ROIs, highlighting the importance of understanding the morphological characteristics of tau. Our heatmaps, which visualized the regions driving the classification of AD and CN using deep learning on tau PET images, showed a distribution pattern similar to group differences in tau deposition between AD and CN assessed using voxel-wise analysis in SPM12. This finding indicates that the deep learning classifier used the morphological characteristics of the tau distribution for classifying AD from CN. In particular, the heatmaps show that the hippocampus, parahippocampal gyrus, thalamus and fusiform gyrus were primarily used to classify AD from CN. These results support existing research showing that tau accumulation in memory-related areas plays an important role in the development of AD [[33](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR33), [34](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR34)].

Early, accurate and efficient diagnosis of AD is important for initiation of effective treatment. Prognostic prediction of the likelihood of conversion of MCI to AD plays a significant role in therapeutic development and ultimately will be important for effective patient care. Thus, the CN vs. AD classifier was used to generate a score showing whether the tau distribution in MCI participants was similar or different from that seen in AD. When the AD probability score generated by the classifier was high, suggesting high similarity to AD, the MCI participants generally had the characteristic tau morphology seen in AD. In addition, we assessed applied this method to both EMCI and LMCI participants. Pearson correlation coefficients between AD probability scores and bilateral mean of SUVR in the medial temporal lobe were R = 0.43 for EMCI and R = 0.49 for LMCI. These findings indicate that the tau deposition difference between the lower 5% and upper 95% of LMCI participants was 7.1% more than the difference between the lower 5% and upper 95% of EMCI participants. Thus, the classifier determined that the tau deposition of LMCI participants is more similar to those seen in AD than that of EMCI participants. This is in line with numerous reports of biomarkers in late MCI where there is considerable overlap with early stage AD pathology [[18](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0#ref-CR18)].

## Conclusion

Deep learning can be used to classify tau PET images from AD patients versus controls. Furthermore, this classifier can score the tau distribution by its similarity to AD when applied to scans from older individuals with MCI. A deep learning derived AD-like tau deposition pattern may be useful for early detection of disease during the prodromal or possibly even preclinical stages of AD on an individual basis. Advances in predictive modeling are needed to develop accurate precision medicine tools for AD and related neurodegenerative disorders, and further developments can be expected with inclusion of multi-modality data sets and larger samples.

<br/>**조태호** (인디애나대학교 의과대학)

[![Github](https://img.shields.io/badge/github-taehojo-yellowgreen)](https://github.com/taehojo)
[![Facebook](https://img.shields.io/badge/facebook-taehojo-blue)](https://facebook.com/taehojo)
[![IU](https://img.shields.io/badge/IU-medicine-red)](https://medicine.iu.edu/faculty/41882/jo-taeho)
<br/>

<br/>

> 이 글은 2020년 12월 28일, BMC Bioinformatics에 게재된  **Taeho Jo et al. Deep learning detection of informative features in tau PET for Alzheimer’s disease classification** 논문을 옮긴 것입니다. 
>
> <details>
> <summary>논문 인용 하기</summary>
> <br>
> Jo, Taeho, et al. "Deep learning detection of informative features in tau PET for Alzheimer’s disease classification." BMC bioinformatics 21.21 (2020): 1-13.
> </details>

[논문 바로가기](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03848-0)

<br/>