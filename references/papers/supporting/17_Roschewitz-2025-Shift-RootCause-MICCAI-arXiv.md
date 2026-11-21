# Automatic dataset shift identification to support root cause analysis of AI performance drift

**Authors:** M. Roschewitz, R. Mehta, C. Jones, B. Glocker

**Venue:** MICCAI 2025 (top conference), arXiv:2411.07940 — Root-cause concept, NOT Q1 journal count

*Source PDF: `17_Roschewitz-2025-Shift-RootCause-MICCAI-arXiv.pdf`*

*Converted to markdown following the Selected_Papers strategy (full-text extraction, page-ordered). Verify title/authors/year against publisher before citing.*

---

## Page 1 / 17

arXiv:2411.07940v3 [cs.AI] 19 Jun 2025
Automatic dataset shift identification to support
safe deployment of medical imaging AI
Mélanie Roschewitz, Raghav Mehta, Charles Jones, Ben Glocker
Imperial College London
Abstract
ShiftsindatadistributioncansubstantiallyharmtheperformanceofclinicalAImodelsandleadtomisdiagnosis.
Hence, various methods have been developed to detect the presence of such shifts at deployment time. However,
the root causes of dataset shifts are diverse, and the choice of shift mitigation strategies is highly dependent on
the precise type of shift encountered at test time. As such,detecting test-time dataset shift is not sufficient: preciselyidentifying which type of shift has occurred is critical. In this work, we propose the first unsupervised dataset
shift identification framework for imaging datasets, effectively distinguishing between prevalence shift (caused by
a change in the label distribution), covariate shift (caused by a change in input characteristics) and mixed shifts
(simultaneous prevalence and covariate shifts). We discuss the importance of self-supervised encoders for detecting subtle covariate shifts and propose a novel shift detector leveraging both self-supervised encoders and task
modeloutputsforimprovedshiftdetection. Weshowtheeffectivenessoftheproposedshiftidentificationframework
across three different imaging modalities (chest radiography, digital mammography, and retinal fundus images) on
five types of real-world dataset shifts using five large publicly available datasets. Code is publicly available at
https://github.com/biomedia-mira/shift_identification.
1 Introduction
Machine learning models are notoriously sensitive to changes in the input data distribution, a phenomenon commonly referred to as dataset shift [44]. This is particularly problematic in clinical settings, where dataset shift is a
common occurrence and may arise from various factors [4]. Changes in the frequency of disease positives over
time or across geographical regions causeprevalence shift[15,27]. The use of different acquisition protocols or
scanners [35,37,41], or a change in patient demographics [2,42] can induce shifts in image characteristics, known
as covariate shift. We illustrate examples of real-world shifts in Fig. 1. Dataset shift can dramatically affect AI
performance of AI and may lead to clinical errors such as misdiagnosis [13,33,34]. It is hence crucial to implement
safeguards allowing not only effective detection of thepresence of shifts, but importantly, reliableidentification of
the root causes.
Dataset shifts can be detected at deployment time by using statistical testing to compare the distributions of incoming test data to the distribution of the reference data (representative of the data used to validate the deployed AI
model). Significant progress has been made in this field where state-of-the-art methods can detect various types
of real-world shifts [12,23,24,30]. Shifts between test and reference data can either be detected at the output level
(by comparing distributions of model outputs), or at the input level (by comparing low-dimensional feature representations of input images) [30]. In this work, we show that different types of shifts require different shift detection
1

## Page 2 / 17

Acquisition shift (e.g. scanner shift) Subpopulation shift (e.g. gender shift)
Male Sick
 Healthy
Sick
Sick
Healthy
Healthy
Sick
 Healthy
Female
COVARIATE SHIFT
Scanner A
PREVALENCE SHIFT
SOURCETARGET
Female
Male
Male
Female
 Female
 Male
Scanner A
 Scanner A
 Scanner B
Scanner B
Scanner B
Scanner C
Scanner A
Figure 1: Examples of dataset shifts in medical imaging. Reliably detecting and identifying the nature of the
shift is crucial to enable the safe deployment of machine learning systems applications. In this work, we propose
the first shiftidentification framework able to reliably detect and classify any detected shift as (i) prevalence, (ii)
covariate or (iii) mixed prevalence and covariate shift, for imaging datasets.
approaches. On the one hand, comparing model output distributions allows for the reliable detection of shifts directly related to the downstream task, such as changes in prevalence. On the other hand, we show that, for shifts
orthogonaltothedownstreamtask, suchaschangesinimageacquisitionprotocols, comparingoutputdistributions
is not sufficient. For such shifts, test and reference data need to be compared at the input level using rich feature
representations. We demonstrate that self-supervised neural network image encoders, trained without using any
task-specific annotations, yield excellent low-dimensional feature representations for shift detection.
While detecting dataset shifts is important, it is insufficient for the safe deployment of AI. Besides knowing that
there is a problem, we need to be able toidentify the precise type of shift to take the necessary actions, implement
preventive measures, and safeguard against harm caused by AI errors. Indeed, many domain adaptation techniques are shift-specific: applying the wrong mitigation technique may, in the best case, be ineffective in resolving
the shift or, in the worst case, severely harm model performance or calibration. For example, prevalence shifts can
often be mitigated with lightweight output recalibration techniques [1,39], but these rely on the assumption that no
other types of shift are present. Applying such label shift adaptation methods when the shift is actually caused by
covariate shift may drastically degrade model calibration and clinical metrics. In contrast, covariate shifts require
more advanced domain adaptation techniques or model finetuning [21,40,46]. For example, image-harmonisation
techniques (e.g. [21]) or automatic correction methods (e.g. [31]) effectively mitigate effects of acquisition shifts on
model performance but will fail in the case of prevalence shift. The difficulty lies in the fact that a change in image
characteristics may cause similar changes in the distribution over model outputs as a change in disease prevalence [31], and determining the cause of an observed shift can be challenging. Despite its importance, automatic
dataset shift identification has remained an open problem.
In this work, we address this issue by proposing a dataset shiftidentification framework capable of identifying the
root cause of the underlying shift, effectively separating (i) prevalence shift, (ii) covariate shift and (iii) mixed shift
(both prevalence and covariate shifts). To the best of our knowledge, this is the first framework able to identify the
type of test-time shifts in an unsupervised manner for imaging data, beyond solely detecting shifts. An in-depth
evaluation across three different clinical applications (chest radiography, digital mammography, and retinal fundus
images)onfivetypesofreal-worlddatasetshiftsdemonstratesthatourframeworkaccuratelydistinguishesbetween
prevalence shifts, covariate shifts, and mixed shifts across various scenarios.
2

## Page 3 / 17

2 Background
2.1 Definitions of prevalence and covariate shift
Formally,let X denotetheinputimageand Y denotethetarget(e.g. diseaselabel). Labelshift (orprevalenceshift)
occurswhenlabeldistributionchangesacrossdomains,i.e. Pref (Y ) ̸= Ptest(Y ),whiletheconditionaldistributions,
i.e. Pref (Y |X) = Ptest(Y |X) are preserved, where Pref and Ptest denote distributions on reference and target
domains respectively. Conversely,covariate shiftoccurs whenPref (X) ̸= Ptest(X), while conditional distributions
are preserved [29]. Acquisition and subpopulation shifts are cases of covariate shifts as they directly affect image
appearance.
2.2 Dataset shift detection methods
Several paradigms have been proposed for dataset shift detection. The simplest method to implement consists of
comparing distributions of a classifier’s outputs between the reference and test domain, proposed by Rabanser et
al. [30], and referred to asBlack Box Shift Detection(BBSD). In detail, softmax model outputs are collected for all
samples in the reference and test sets. Then, for each class, a separate univariate Kolmogorov-Smirnov (K-S) test
is run to determine if the class-wise predicted probabilities distributions differ between reference and test domain,
the overall significance of the shift is then determined after applying Bonferroni correction [11] for multiple testing.
In the same study, Rabanser et al. [30] also proposed another type of shift detector where reference and test data
input distribution are compared using a feature-based approach. In this test, the input sample (image) first gets
projected to a smaller dimension, e.g. through a pretrained neural network encoder and the shift is then measured
using theMaximum Mean Discrepancypermutation test originally proposed by Gretton et al. [16]. The Maximum
Mean Discrepancy measures the distance between two distributionsP and Q based on the distance between their
mean embeddings. An unbiased estimate of the square of the MMD statistic can be computed via:
\M M D
2
= 1
m2 − m
mX
i=1
mX
j̸=i
κ(zi, zj) + 1
n2 − n
nX
i=1
nX
j̸=i
κ(z′
i, z′
j) − 2
mn
mX
i=1
nX
j=1
κ(zi, z′
j), (1)
where {zi}m
i=1 ∼ P,and {z′
i}n
i=1 ∼ Qand κisakernelovertheembeddingspace. Thep-valuecanthenbeobtained
usingapermutationtest. InRabanseretal.[30], theyproposedtousetheRBFkernel κ(z, ˜z) = e− 1
2σ ||z−˜z||2
, setting
σ as the median distance between all samples. An alternative way would be to explicitly learn the kernel, also
known as ‘deep kernels’ [25]. The disadvantage of this approach is that the kernel needs to be learned for every
single test set for which one wishes to run shift detection. In this work, we use the RBF kernel.
Another existing shift detection approach consists of training a domain classifier to classify samples between the
reference and test domains [6,19,26] and using the accuracy of this classifier as a proxy for measuring distances
between distributions. One drawback of this approach is its high computational cost: for every test set, a new
domain classifier must be trained, which is highly impractical in continuous monitoring scenarios. Hence, we here
focus on output-based (BBSD) and feature-based (MMD) shift detection methods, as these do not require training
of additional models at test-time.
In the medical imaging domain, all these shift detectors were benchmarked by Koch et al. [23] for their ability to
detect a large variety of real-world shifts in the context of diabetic retinopathy grading models. Both the domain
classifierandtheBBSDapproachwereshowntosuccessfullydetectthetestedshifts,theirsensitivitydependingon
thenumberoftestsamples,whereasMMD-basedtestswerefoundtobelessaccurateatdetectingshiftscompared
3

## Page 4 / 17

to BBSDs and domain classifiers. Given these results, and for computational efficiency, we herein focus on shift
detection methods that do not require the training of additional models at test-time.
3 Methods and experimental setup
3.1 Dataset shift identification pipeline
Estimate 
prevalence on 
test set
.
30% 
positives
Resample 
reference set
.
30% 
positives
Test set
Prevalence
-
adjusted 
reference
Significant 
differences 
PREVALENCE 
SHIFT ONLY
No 
difference
If differences 
disappear after 
prevalence 
adjustment
PREVALENCE 
SHIFT 
PRESENT
Else
COVARIATE SHIFT 
ONLY
B. SHIFT IDENTIFICATION
Reference set
Test set
Encoder
Task 
model
Encoder
Task 
model
Task 
model
P(y) 
= 0.8
P(y) 
= 0.9
Test output distribution
Reference feature distribution
A. SHIFT DETECTION
No differences
Significant differences
SHIFT DETECTED
PIPELINE OVERVIEW
Unlabelled test set 
Model decisions
No shift
Shift detected
A. SHIFT 
DETECTION
B. SHIFT 
IDENTIFICATION
NO SHIFT
PREVALENCE + 
COVARIATE SHIFT
COVARIATE SHIFT
PREVALENCE SHIFT
Annotated reference set
Test feature distribution
Reference output distribution
COMPARE
HEALTHY
COMPARE
NO SHIFT
1
2
3
4
5
COVARIATE 
SHIFT 
PRESENT
Original 
reference
Prev.
-
adjusted 
reference
Test
Test
COMPARE
COMPARE
6
PREVALENCE + 
COVARIATE SHIFT
Figure 2:Overview of the proposed dataset shift identification pipeline. We leverage both task model outputs
andfeaturesfromself-supervisedencodersfordetectingandidentifyingdatasetshifts. Contrarilytopreviousworks,
we do not simply detect thepresence of shifts but also add a second step able toidentify the nature of the shift.
Our method effectively separates cases of (i) prevalence shift (a change in label distribution), (ii) covariate shift (a
change in image characteristics) and (iii) covariate and prevalence shift (both).
Here, we propose a framework for identifying whether dataset shift is caused by prevalence shift, by covariate shift
or by a mix of both. The approach is divided into two stages, as depicted in Fig. 2:
1. Standard dataset shift detectionto separate the ‘shift’ from the ‘no shift’ cases (Fig. 2, A). For this step, we
use a dual detection approach, combining signals from task model outputs and features from self-supervised
(SSL) encoders to detect shifts. In detail, we first independently run the BBSD and the MMD shift detection
tests; this yieldsC p-values for the BBSD test (one per class) and one p-value for the MMD permutation test.
We then apply Bonferroni correction on theC + 1 p-values to get the overall significance.
4

## Page 5 / 17

2. If a shift is detected in the first step, we then proceed toshift identification. This process starts with estimating the prevalence in the test set (Fig. 2, B.3). For this, we can leverage the prevalence shift adaptation
literature, where various methods have been proposed to estimate the density ratio:
w := Pref (Y )
Ptest(Y ) (2)
with w ∈ RC [1,32,39]. Here, we use the state-of-the-art ‘class probability matching with calibrated network’
(CPMCN) method by Wen et al. [39] to estimate this ratio and the test set prevalence. In CPMCN, the density
ratio is estimated by:
ˆw := arg min
w∈RC
CX
i=1

ˆPref (Y = i) − 1
m
X
x∈Dt
ˆp(i|x)PC
j wj ˆp(j|x)

2
(3)
with mthe number of samples in the test setDt, ˆPref (Y = i)is the empirical proportion of samples with class
i in the reference set, andˆp(i|x) is the probability predicted by the model for classi given sample x. Given
this estimated ratioˆw, we can easily recover the estimated label distribution on the test domain
ˆPtest(Y = i) = ˆwi · ˆPref (Y = i), ∀i ∈ C (4)
We follow this method to estimate the label distribution in the test set in our shift identification module. Next,
weresamplethereferencesettomatchthisestimatedprevalence(Fig.2, B.4). Wethenfirstcomparefeature
distributions between prevalence-adjusted reference and test set (Fig. 2, B.5): if differences are no longer
significant after adjusting the prevalence, the shift is attributed to prevalence shift. Conversely, if differences
persist after adjusting the prevalence, then covariate shift is necessarily present. In this case, we compare
model output distributions with BBSD to determine whether prevalence shift is also present (Fig. 2, B.6).
Precisely, if there were significant differences in model output distributions before adjusting the prevalence,
but this shift disappears after adjusting the prevalence, we know that prevalence shift is also responsible for
the observed shift, in this case we conclude that the observed shift is a case of mixed shift (prevalence +
covariate shift). Else, we conclude that the shift is attributed to covariate shift only.
3.2 Datasets
We evaluate our methods on four different datasets covering three different imaging modalities: (i) chest radiography, (ii) mammography, and (iii) fundus images.
For chest radiography, we use two public datasets for our analysis: theRSNA Pneumoniadataset [36], a subset of the NIH Chest-Xray8 dataset [38] manually relabelled by expert radiologists for presence of pneumonia-like
opacities. We use the original metadata from the NIH Chest-Xray8 dataset [38] to retrieve patient gender. We also
use PadChest [3] a larger dataset containing multiple disease labels extracted from radiology reports. We here
focus on the pneumonia label. Importantly, this dataset contains important metadata such as scanner information
or the gender of the patient, allowing us to generate a wide range of shifts. For mammography, we use theEMBED dataset [20] a large mammogram dataset collected in the US, on six different scanners. Finally, for fundus
imaging, we createRETINA a multi-domain dataset by combining three different public datasets: the Kaggle Diabetic Retinopathy Detection [10] dataset, the Kaggle Aptos Blindness Detection dataset [22] and the Messidor-v2
dataset [8]. These datasets cover different regions of the world (India, France, US) but also with varying image acquisition devices: images from the Messidor-v2 are high-quality images, while many images in the Kaggle datasets
5

## Page 6 / 17

are of lower quality (including phone pictures). Creating this multi-centre dataset allows us to simulate various
domain shifts by varying the proportion of data from each ‘site’ (original dataset source). The task of interest for
fundus images here is binary diabetic retinopathy (DR) classification for fundus images, where we classify images
between referable DR (grades 2,3,4) and healthy/non-referable DR (grades 0,1).
3.3 Shift generation details
For every dataset, we study different types of shifts at different levels of intensity. To study prevalence shift detection, we associate each dataset with a downstream task. For chest radiography datasets we focus on pneumonia
detection, for mammography on breast density assessment (4 classes), and for retinal images on binary diabetic
retinopathy classification. We simulate various levels of prevalence shift by resampling the test set according to
specific label distributions. Then, we study various types of covariate shifts. For PadChest, we study gender shifts
by varying the proportion of female patients in the test set. Moreover, PadChest contains scans acquired with two
scanners, ‘Phillips’ (40%) and ‘Imaging’ (60%). This allows the simulation of different levels of acquisition shift by
varying the proportion of Phillips scans in the test set. Similarly, for EMBED [20], we study acquisition shift by
varying the distribution of scanners in the test set. This dataset offers a complementary view to PadChest, with
a multi-class task of interest and providing even more flexibility for simulating diverse acquisition shifts (six scanners). Note that, in EMBED, each exam comprises four mammograms (left/right breasts and MLO/CC views). We
excluded all exams that did not contain exactly four images, kept exactly one exam per patient, and ensured that
test set sampling was done at the exam level. Finally, for the RETINA dataset, we simulate covariate shifts by
varying the proportion of samples coming from each underlying dataset (Aptos, Kaggle DR and Messidor).
3.4 Implementation details
Toevaluatetheshiftidentificationaccuracyoftheproposedframework,wefollowstandardevaluationpracticesfrom
the dataset shift detection literature. Specifically, we repeat the following process 200 times: (i) sample a subset of
size Ntest from the test split according to the shift of interest and sample the reference set from the validation split,
(ii) run the shift identification test, and (iii) record whether the shift is correctly identified. The final shift identification
accuracy is computed as the proportion of times the shift is correctly identified out of all the bootstrap samples.
For both chest radiography datasets, we usedNref = 2, 000 images andNtest ∈ {100, 250, 500, 1000} images. For
the RETINA dataset, we usedNref = 1000 images and Ntest ∈ { 100, 250, 1000} images. The case of EMBED is
slightlydifferentasweneedtoensurethatimagesbelongingtothesameexamarealwayssampledsimultaneously.
Hence, for this dataset sampling of all the reference sets, shifted test sets (and permutations in the MMD test1) is
done at the exam level and not at the image level. We assumedNexams,ref = 1000 exams (i.e4000 images) and
Nexams,test ∈ {50, 100, 250}.
All task models and encoders used in this study are ResNet-50 [18]. For each dataset, we use the task model for
the BBSD test. For the MMD detection test, we first extract the embeddings from the last layer of the encoder, then
project them onto the 32-first principle components and then use the RBF kernel. We compare various encoders
for feature extraction, in particular, encoders trained in a self-supervised manner. Self-supervised (SSL) encoders
were trained using the SimCLR [5] objective. Additionally, for the RETINA dataset, we compare with RetFound [45]
a publicly available self-supervised foundation model trained with the Masked Auto Encoder [17] objective on a
large set of retinal images, based on a vision transformer architecture [9].
1Keeping the underlying structure of the data during permutation tests is important to avoid inflated type I error [7].
6

## Page 7 / 17

4 Results
4.1 Different shifts require different detectors
Priortodivingintoshift identification,wefirstinvestigatewhichtypesofshiftsaresuccessfully detectedbyprominent
dataset shift detection methods. We compare two families of shift detectors: model output-based (BBSD) and
feature-based detectors (MMD). We additionally test a dual approach that combines both approaches for improved
shift detection (‘Duo’, as described in Section 3.1).
For feature-based shift detection, any pretrained network could be used as feature extractor. A perhaps obvious
choice is to simply use the encoder from the task-specific classification model. However, this may not be the
best choice as learned features will be heavily skewed towards encoding characteristics specifically relevant to
that task as opposed to encoding more generic image representations sensitive to data distribution changes [43].
Hence, we here explore the potential of SSL image encoders for shift detection. Indeed, encoders trained in a
self-supervised manner, i.e. without any labels, learn fine-grained representations effectively summarising all the
information encoded in a given image, resulting in ideal candidates for generic shift detection. We compare the
performance of feature-based shift detection for five different encoders (using the features obtained from the last
layer of the neural network encoder). We compare: (i)Random a ResNet-50 encoder with random weights; (ii)
Supervised ImageNetwhere features are extracted using a ResNet encoder trained to perform classification on
natural images from the ImageNet dataset; (iii)Task model, the ResNet encoder from the task model used to
perform the downstream classification task; (iv)SSL ImageNeta self-supervised encoder trained on ImageNet
data only; (v)SSL Modality Specifica self-supervised encoder trained on the same modality as the test datasets.
Additionally, for the RETINA dataset, we include a comparison using theRetFound foundation model as a feature
extractor [45].
Output- and feature-based detectors detect different shifts.Figs. 3 and 4 show the shift detection rates for
everydataset-shiftcombination. Acrossalldatasets, aclearpatternappears. Forprevalenceshifts(Fig.3), outputbased shift detection performs significantly better than all feature-based tests. This is intuitive as the shift is directly
related to the downstream prediction task. A change in prevalence should directly be reflected by a change in the
distribution of task model outputs. For covariate shifts, results are very different, regardless of whether we look at
acquisition or subpopulation shifts (see Fig. 4). For these shifts, most feature-based detectors perform substantially better than output-based detectors. Output-based shift detection fails to detect gender shifts on radiography
datasets (less than 5% of shifts detected) and performs substantially worse than feature-based tests on acquisition
shifts across all levels of shifts and datasets. For example, on EMBED with 250 test exams, the output-based
shift detector only detects 25% of acquisition shifts, whereas feature-based detectors detect at least 80% of shifts
(except for the random encoder).
Self-supervisedencoderscandetectsubtlecovariateshifts. Ourresultsshowthattheeffectivenessoffeaturebased dataset shift detectors is highly dependent on the choice of the encoder used to extract the features (see
differences between feature-based detection rates in shades of blue in Fig. 4). The results indicate that for optimal
detection of more subtle shifts (e.g. gender shifts), it is important to use an encoder trained in a self-supervised
manner. The task model, for example, under-performs for gender shift detection on chest radiography datasets,
and so does the encoder trained in a supervised manner on ImageNet data, particularly visible for milder shifts and
smaller test set sizes. For example, both supervised encoders fail to detect the shift from 44% to 25% of females in
the RSNA dataset, even with large test sizes (Fig. 4 top row). Encoders trained in a self-supervised manner (SSL
7

## Page 8 / 17

100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * ** **
* *
* *
Shifted prevalence: 10% (-13%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * * * * * * * * * *
Shifted prevalence: 50% (+27%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * ** * * * * * * * * * * * * * * *
Shifted prevalence: 80% (+57%)
RSNA Pneumonia - Original prevalence : 23%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* *
* * * * * * * * *
Prevalence: 15% (+11%)
100 250 1000
Number of test images
0
25
50
75
100
 * *
* *
* *
Prevalence: 20% (+16%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * *
* *
* *
Prevalence: 30% (+26%)
PadChest - Original prevalence: 4%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100Detected shift (%)
* *
* *
* *
Density class distribution: 0%, 50%, 50%, 0%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100
 * *
* *
* *
Density class distribution: 15%, 35%, 35%, 15%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100
 * ** * * *
Density class distribution: 10%, 20%, 60%, 10%
EMBED - Original density distribution: 7%, 37%, 47%, 7%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * * * * * *
Prevalence shift 50% (-28%)
100 250 1000
Number of test images
0
25
50
75
100
** *
* *
* *
Prevalence shift 65% (-13%)
100 250 1000
Number of test images
0
25
50
75
100
* *
* * * *
Prevalence shift 100% (+22%)
RETINA - Original disease prevalence: 0.78
MMD (Random encoder features)
MMD (Supervised ImageNet features)
MMD (Task model features)
MMD (SSL ImageNet features)
MMD (SSL Modality Speciﬁc features)
MMD (Retfound features)
Duo (model output + SSL ImageNet features)
Shift detector
BBSD (task model)
Figure 3: Prevalence shift: shift detectors comparison. We report the shift detection rate over 200 bootstrap
samples. Results show that output-based detection is best for detecting prevalence shifts. For each combination
of test set size and type of shift, the detector with the highest detection rate, as well as all detectors not significantly
different from the best, are denoted with an asterisk∗ (where significance is measured by Fisher’s exact test, at
level .05, with Bonferonni correction applied for multiple testing).
8

## Page 9 / 17

100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * * *
*
*
Proportion of females: 25% (-19%)
100 250 1000
Number of test images
0
25
50
75
100
*
* * * *
Proportion of females: 75% (+31%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * * * * * * * * * *
Proportion of females: 100% (+56%)
RSNA Pneumonia - Gender shift
Original proportion of females: 44%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * * *
*
*
Proportion of females: 25% (-26%)
100 250 1000
Number of test images
0
25
50
75
100
 * * *
*
*
Proportion of females: 75% (+24%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * * * ** * * * * *
Proportion of females: 100% (+49%)
PadChest - Gender shift
Original proportion of females: 51%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * ** *
*
* *
Proportion of Phillips: 25% (-17%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * * * * ** * * * * * * * * *
Proportion of Phillips: 75% (+33%)
100 250 1000
Number of test images
0
25
50
75
100
 * * * * * * ** * * * * * * * * * * * * *
Proportion of Phillips: 100% (+58%)
PadChest - Acquisition shift
Original proportion of Phillips: 42%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100Detected shift (%)
* * * *
* * * * *
Scanner distribution: 55%,0%,10%,10%,15%,10%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100
*
* * * * *
Scanner distribution: 50%,0%,0%,20%,20%,10%
50 100 250
Number of test exams (each with 4 images)
0
25
50
75
100
 * * * * * * * * * * * * * *
Scanner distribution: 33%,2%,20%,15%,20%,10%
EMBED - Acquisition shift
Original scanner distribution: 79%,0%,5%,4%,7%,5%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
*** * * ***** ********
Domain distribution [0.2, 0.2, 0.6]
100 250 1000
Number of test images
0
25
50
75
100
 * *** * * **** ********
Domain distribution [0.05, 0.3, 0.65]
100 250 1000
Number of test images
0
25
50
75
100
 ****** ******** ********
Domain distribution [0.3, 0.3, 0.4]
RETINA - Acquisition shift
Original domain distribution: [0.04, 0.09, 0.87]
MMD (Random encoder features)
MMD (Supervised ImageNet features)
MMD (Task model features)
MMD (SSL ImageNet features)
MMD (SSL Modality Speciﬁc features)
MMD (Retfound features)
Duo (model output + SSL ImageNet features)
Shift detector
BBSD (task model)
Figure 4:Covariate shifts: shift detectors comparison. We studied two sub-types of covariate shifts: subpopulation shift (top two rows) and acquisition shift (bottom three rows). We report the shift detection rate over 200
bootstrap samples. Results show that feature-based detection is best for this type of shift. For each combination
of test set size and type of shift, the detector with the highest detection rate, as well as all detectors not significantly
different from the best, are denoted with an asterisk∗ (where significance is measured by Fisher’s exact test, at
level .05, with Bonferonni correction applied for multiple testing).
9

## Page 10 / 17

100 250 1000
Number of test images
0
25
50
75
100Wrongly detected shift (%)
RETINA
100 250 1000
Number of test images
0
25
50
75
100
 RSNA Pneumonia
100 250 1000
Number of test images
0
25
50
75
100
 PadChest
50 100 250
Number of test exams (4 images/exam)
0
25
50
75
100
 EMBED
Output-based (task model) Feature-based (SSL ImageNet) Duo (model output + SSL ImageNet features)Shift detector Output-based (task model) Feature-based (SSL ImageNet) Duo (model output + SSL ImageNet features)
Figure 5:False detection rate: shift detectors comparison. Percentage of shift detected when resampling the
test set without shift.
ImageNetandSSLModalitySpecific)detectgendershiftswithsignificantlyhighersensitivity, exhibitinganaverage
sensitivity of 85% across all gender shifts with a test set size of 250 and 100% for a test set size of 1000. Similarly,
for acquisition shifts, results show that SSL encoders offer substantially better detection rates than their supervised
counterparts. The SSL model trained on ImageNet data was particularly effective and, in some cases, even better
than the modality-specific SSL models for PadChest and EMBED, especially for subtle shifts and in the low test
data regime. On the RETINA dataset, self-supervised encoders also all outperform their supervised counterparts,
with no significant differences between self-supervised encoders.
Combining output- and feature-based detection.Building on the previous findings, we evaluate a dual detection approach combining responses from output-based and feature-based shift detectors, using self-supervised
features for more robust shift detection. Results in Figs. 3 and 4 demonstrate that the proposed ‘Duo’ detector
(in orange) performs best overall across shifts and datasets, regardless of the shift severity. For example, with
the duo detector and test set size of 1000 samples, the detection accuracy is>80% for nearly every shift, across
all datasets. On the contrary, the other detectors respectively fail either on prevalence shift (with feature-based
detectors detecting less than 50% prevalence shift cases averaged over datasets) or on covariate shift (where
output-based detection detects less than 5% of gender shifts for radiography and less than 25% of acquisition
shifts for EMBED). We employ this dual approach for the detection module of our shift identification pipeline.
For completeness, in Fig. 5 we report the false shift detection rate of shift detectors when generating test sets
without shifts. We verify that the false-positive rates of the output-based, feature-based and Duo detectors stay
low across datasets and detectors, we find that false detection rates hover around the expected type-I error of the
statistical tests 5% for most datasets and detection methods. The only exception is EMBED, where false positive
ratesarealittlehigherthanexpectedfortheoutput-based(andhenceDuo)detector. Thisisduetothefactthat-for
this highly imbalanced classification problem - small randomly resampled test sets may lead to small differences in
theobservedcumulativedistributionofpredictions, evenintheabsenceofashiftinthetestsetgenerationprocess.
4.2 Shift identification performance
To perform shift identification, we leverage the fact that output-based and feature-based shift measures detect
different types of shifts to precisely identify the shift present in the test dataset, following the decision logic detailed
in Section 3.1 and illustrated in Fig. 2.
In-depth evaluation results in Figs. 6 and 7 demonstrate that our novel shift identification framework is capable of
distinguishing between prevalence shifts, covariate shifts and mixed shifts with high accuracy across all datasets
10

## Page 11 / 17

and types of shifts. Overall, more subtle shifts are best detected with larger test sets whereas larger shifts can be
detected with smaller test sets. For prevalence shifts, the average shift identification rate, across datasets and shift
levels, is 85% with 500 test images and reaches 95% with 1000 test images (Fig. 6, top two rows).
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Females: 75% (+31%)
Disease prev: 10% (-13%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 75% (+31%)
Disease prev: 50% (+27%)
100 250 500 1000
0
25
50
75
100
Females: 100% (+56%)
Disease prev: 10% (-13%)
RSNAPneumonia - Gender + Prevalence shift
Original proportion of females: 44%
Original disease prevalence: 23%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Females: 25% (-26%)
Disease prev.: 15% (+11%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 75% (+24%)
Disease prev.: 15% (+11%)
100 250 500 1000
0
25
50
75
100
Females: 100% (+49%)
Disease prev.: 15% (+11%)
PadChest - Gender + Prevalence shift
Original proportion of females: 51%
Original disease prevalence: 4%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Proportion of Phillips scans:
75% (+71%)
Disease prevalence:
15% (+11%)
100 250 500 1000
Number of test images
0
25
50
75
100
Proportion of Phillips scans:
75% (+71%)
Disease prevalence:
25% (+21%)
100 250 500 1000
0
25
50
75
100
Proportion of Phillips scans
100% (+96%)
Disease prevalence:
25% (+21%)
PadChest - Acquisition + Prevalence shift
Original proportion of Phillips scans: 42%
Original disease prevalence: 4%
50 100 250
0
25
50
75
100Identiﬁed shift (%)
Scanner distribution:
55%,0%,10%,10%,15%,10%
Density distribution:
0%, 50%, 50%, 0%
50 100 250
Number of test exams (4 images/exam)
0
25
50
75
100
Scanner distribution:
50%,0%,0%,20%,20%,10%
Density distribution:
0%, 50%, 50%, 0%
50 100 250
0
25
50
75
100
Scanner distribution:
33%,2%,20%,15%,20%,10%
Density distribution:
0%, 50%, 50%, 0%
EMBED - Acquisition + Prevalence shift
Original scanner distribution: 79%, 0%, 5%, 4%, 7%, 5%
Original density distribution: 7%, 37%, 47%, 7%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Site distribution:
20%, 20%, 60%
Disease prevalence: 
50% (-28%)
100 250 500 1000
Number of test images
0
25
50
75
100
Site distribution:
5%, 30%, 65%
Disease prevalence: 
50% (-28%)
100 250 500 1000
0
25
50
75
100
Site distribution:
30%, 30%, 40%
Disease prevalence: 
50% (-28%)
RETINA - Acquisition + Prevalence shift
Original site distribution: 4%, 9%, 87%
Original disease prevalence: 78%
Prevalence shift Covariate only Covariate + Prevalence Correct IncorrectCovariate + Prevalence
Figure7: Shiftidentificationaccuracy: mixedcovariateandprevalenceshifts . (Toprow)depictsmixedgender
and prevalence shift, (bottom two rows) show mixed acquisition and prevalence shift. Across all datasets, the
shift identification framework is able to successfully detect and identify presence of both shifts with high accuracy.
Identification accuracy is computed over 200 bootstrap samples.
Similarly, for covariate shifts caused by acquisition shifts, when using a test set size of 500 images (and 250 test
exams on EMBED), the identification accuracy is greater than 80% for any shift level and dataset, with an average
identification accuracy of 89% across datasets and shift levels (Fig. 6 bottom two rows). For the RETINA dataset,
identification rates already reach 100% with a test set as small as 250 images. When the covariate shift is induced
by gender shifts, covariate shift is detected with identification accuracies greater than 90% for both PadChest and
RSNA Pneumonia, with a test set size of 500 images, for all but one shift level (Fig. 6). Results in Fig. 7, show that
the framework is also able to accurately distinguish between cases of covariate shift only and cases of covariate
and prevalence shifts. For these mixed shifts, shifts are identified with increasingly high accuracy as the test set
11

## Page 12 / 17

100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Prevalence:
10% (-13%)
100 250 500 1000
Number of test images
0
25
50
75
100
Prevalence:
50% (+27%)
100 250 500 1000
0
25
50
75
100
Prevalence:
80% (+57%)
RSNAPneumonia - Original prevalence: 23%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Prevalence:
15% (+11%)
100 250 500 1000
Number of test images
0
25
50
75
100
Prevalence:
20% (+16%)
100 250 500 1000
0
25
50
75
100
Prevalence:
30% (+26%)
PadChest - Original prevalence: 4%
50 100 250
0
25
50
75
100Identiﬁed shift (%)
Prevalence:
0%,50%,50%,0%
50 100 250
Number of test exams (4 images/exam)
0
25
50
75
100
Prevalence:
15%,35%,35%,15%
50 100 250
0
25
50
75
100
Prevalence:
10%,20%,60%,10%
EMBED - Original class distribution: 7%,37%,47%,7%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Prevalence:
50% (-28%)
100 250 500 1000
Number of test images
0
25
50
75
100
Prevalence:
65% (-13%)
100 250 500 1000
0
25
50
75
100
Prevalence:
100% (+22%)
RETINA - Original prevalence: 78%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Females: 25% (-19%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 75% (+31%)
100 250 500 1000
0
25
50
75
100
Females: 100% (+56%)
RSNAPneumonia - Gender shift
Original proportion of females: 44%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Females: 25% (-26%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 75% (+24%)
100 250 500 1000
0
25
50
75
100
Females: 100% (+49%)
PadChest - Gender shift
Original proportion of females: 51%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Proportion Phillips scans:
25% (-17%)
100 250 500 1000
Number of test images
0
25
50
75
100
Proportion Phillips scans:
75% (+33%)
100 250 500 1000
0
25
50
75
100
Proportion Phillips scans:
100% (+58%)
PadChest - Acquisition shift
Original proportion of Phillips scans: 42%
50 100 250
0
25
50
75
100Identiﬁed shift (%)
Scanner distribution:
55%,0%,10%,10%,15%,10%
50 100 250
Number of test exams (4 images per exam)
0
25
50
75
100
Scanner distribution:
50%,0%,0%,20%,20%,10%
50 100 250
0
25
50
75
100
Scanner distribution:
33%,2%,20%,15%,20%,10%
EMBED - Acquisition shift
Original scanner distribution: 79%,0%,5%,4%,7%,5%
100 250 500 1000
0
25
50
75
100Identiﬁed shift (%)
Site distribution:
20%,20%,60%
100 250 500 1000
Number of test images
0
25
50
75
100
Site distribution:
5%,30%,65%
100 250 500 1000
0
25
50
75
100
Site distribution:
30%,30%,40%
RETINA - Acquisition shift
Original site distribution: 4%,9%,87%
Prevalence shift Covariate only Covariate + Prevalence Correct IncorrectCovariate + Prevalence
Figure 6:Shift identification accuracy: prevalence shifts (top two rows) and covariate shifts (bottom three
rows). Across all datasets, the shift identification framework is able to successfully detect and identify both prevalenceshiftsandcovariateshiftswithhighaccuracy. Identificationaccuracyiscomputedover200bootstrapsamples.
12

## Page 13 / 17

size increases. With a test set of 1000 images, mixed gender and prevalence shifts are correctly identified as
mixed shifts with an average accuracy of 97% for RSNA Pneumonia and 75% for PadChest. Mixed shifts induced
by acquisition and prevalence shifts are detected with an average 100% accuracy for the RETINA dataset, 77% for
PadChest and 79% for EMBED, across shift levels with 1000 test images. The overall identification accuracy for
mixed shifts across all shifts and datasets is 85% (with 1000 test images).
5 Discussion
By analysing common dataset shift detection paradigms, we find that different types of shifts require different types
of shift detectors. Our analysis also demonstrates the importance of the choice of encoders for feature-based
dataset shift detection. In particular, we find that encoders trained in a self-supervised manner yield features with
substantially higher shift detection power than supervised counterparts. Maybe surprisingly, our results show that
generic encoders trained in a self-supervised manner on natural images (ImageNet) provide highly discriminative
features for medical image dataset shift detection, transportable across datasets. Following these findings, we
evaluate a new dual dataset shift detector, combining shift detection signals from task model outputs tests and
features from self-supervised encoders. This approach outperforms existing shift detectors. The consistency of
shift detection performance across various types of shifts is crucial as the nature of the shift is, by definition,
unknown at test time.
We take this combined approach a step further to perform fine-grained shift identification, showing high accuracy
in correctly identifying the type of shift present in the test set across all modalities, tasks and various levels of
shift intensity. Our results demonstrate the practical value of the shift identification method, which does not require
any training at test time, nor any ground truth labels or annotations on the test domain data. The use of a readily
available self-supervised encoder trained on ImageNet data for feature extraction dispenses us from training any
additional model for shift detection and identification purposes. Combining signals from both feature-based and
model output-based shift detectors yields reliable and consistent detection and identification across all types of
shifts. Importantly, our method not only separates covariate shifts from prevalence shifts but also reliably detects
when both types of shifts are present in the test set, rendering the proposed method applicable to many real-world
deployment scenarios.
In practice, the proposed framework can be used as a continuous monitoring tool for any image-based, clinical AI
model. Similar to other continuous monitoring tools [28], the shift identification framework would run on the stream
of incoming data, collecting test data on a rolling time window. The reference data should be a small dataset
representative of the expected data distribution on which the model has been validated. Our results show that with
at most 2000 reference images, a test set of only 500 images suffices to yield high shift detection and identification
accuracy across all shifts and datasets (> 80%).
In terms of limitations, we note that in the case of covariate shift, on its own, the proposed shift identification frameworkdoesnotallowforamorefine-grainedidentificationoftheoriginofshift,e.g. thedistinctionbetweenpopulation
andacquisitionshifts. Toallowforanevenmoreprecisesub-typeshiftidentification, integratingmetadatastatistics
in the pipeline (e.g. as in multi-modal shift detection pipelines [28]) could complement the framework. Nevertheless, it is important to highlight that relying solely on metadata monitoring only enables the detection of shifts
affecting the specific attributes collected at test time. For example, statistics on patient population, such as age
distribution, gender distribution, may be recorded at deployment time and used to detectsome sub-types of shifts.
Should the metadata not be available at test time, one could use auxiliary models to predict attributes of interest
13

## Page 14 / 17

from images directly (e.g. ethnicity [14]). However, solely relying on collected (or predicted) metadata may not
capture all sources of covariate shifts and will not allow detection of prevalence shift. The proposed framework can
help uncover shifts that are not detectable by means of simply tracking population metadata. Shift detection and
identification can prompt further investigation and inform root cause analysis of AI performance degradation.
Acknowledgments
M.R. is funded by an Imperial College London President’s PhD Scholarship and a Google PhD Fellowship. R.M.
is funded through the European Union’s Horizon Europe research and innovation programme under grant agreement 10108030. C.J. is supported by Microsoft Research and EPSRC through the Microsoft PhD Scholarship
Programme. B.G. acknowledges support from the Royal Academy of Engineering as part of his Kheiron Medical
Technologies/RAEng Research Chair in Safe Deployment of Medical Imaging AI.
Disclosure of interests
B.G. is part-time employee of DeepHealth. No other competing interests.
References
[1] A. Alexandari, A. Kundaje, and A. Shrikumar. Maximum Likelihood with Bias-Corrected Calibration is HardTo-Beat at Label Shift Adaptation. InProceedings of the 37th International Conference on Machine Learning,
pages 222–232. PMLR, Nov. 2020. ISSN: 2640-3498.
[2] Y.Appelman,B.B.vanRijn,M.E.tenHaaf,E.Boersma,andS.A.E.Peters. Sexdifferencesincardiovascular
risk factors and disease prevention.Atherosclerosis, 241(1):211–218, July 2015.
[3] A. Bustos, A. Pertusa, J.-M. Salinas, and M. de la Iglesia-Vayá. PadChest: A large chest x-ray image dataset
with multi-label annotated reports.Medical Image Analysis, 66:101797, Dec. 2020.
[4] D. C. Castro, I. Walker, and B. Glocker. Causality matters in medical imaging.Nature Communications,
11(1):3673, July 2020. Publisher: Nature Publishing Group.
[5] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A Simple Framework for Contrastive Learning of Visual
Representations. In Proceedings of the 37th International Conference on Machine Learning, pages 1597–
1607. PMLR, Nov. 2020. ISSN: 2640-3498.
[6] X. Cheng and A. Cloninger. Classification Logit Two-Sample Testing by Neural Networks for Differentiating
Near Manifold Densities.IEEE Transactions on Information Theory, 68(10):6631–6662, Oct. 2022.
[7] G. A. Churchill and R. W. Doerge. Naive Application of Permutation Testing Leads to Inflated Type I Error
Rates. Genetics, 178(1):609–610, Jan. 2008.
[8] E. Decencière, X. Zhang, G. Cazuguel, B. Lay, B. Cochener, C. Trone, P. Gain, R. Ordonez, P. Massin,
A. Erginay, B. Charton, and J.-C. Klein. Feedback on a publicly distributed image database: the messidor
database. Image Analysis and Stereology, 33(3):231–234, Aug. 2014. Number: 3.
[9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer,
G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale. InInternational Conference on Learning Representations, Oct. 2020.
14

## Page 15 / 17

[10] E. Dugas, J. Jared, and W. Cukierski. Diabetic Retinopathy Detection Kaggle Challenge, 2015.
[11] O.J.Dunn. MultipleComparisonsamongMeans. JournaloftheAmericanStatisticalAssociation ,56(293):52–
64, Mar. 1961. Publisher: Taylor & Francis.
[12] J. Feng, A. Subbaswamy, A. Gossmann, H. Singh, B. Sahiner, M.-O. Kim, G. A. Pennello, N. Petrick, R. Pirracchio, and F. Xia. Designing monitoring strategies for deployed machine learning algorithms: navigating
performativity through a causal lens. InProceedings of the Third Conference on Causal Learning and Reasoning, pages 587–608. PMLR, Mar. 2024.
[13] S. G. Finlayson, A. Subbaswamy, K. Singh, J. Bowers, A. Kupke, J. Zittrain, I. S. Kohane,
and S. Suchi. The Clinician and Dataset Shift in Artificial Intelligence. New England Journal of Medicine , 385(3):283–286, July 2021. Publisher: Massachusetts Medical Society _eprint:
https://www.nejm.org/doi/pdf/10.1056/NEJMc2104626.
[14] J. W. Gichoya, I. Banerjee, A. R. Bhimireddy, J. L. Burns, L. A. Celi, L.-C. Chen, R. Correa, N. Dullerud,
M. Ghassemi, S.-C. Huang, P.-C. Kuo, M. P. Lungren, L. J. Palmer, B. J. Price, S. Purkayastha, A. T. Pyrros,
L. Oakden-Rayner, C. Okechukwu, L. Seyyed-Kalantari, H. Trivedi, R. Wang, Z. Zaiman, and H. Zhang. AI
recognition of patient race in medical imaging: a modelling study.The Lancet Digital Health, 4(6):e406–e414,
June 2022. Publisher: Elsevier.
[15] P. Godau, P. Kalinowski, E. Christodoulou, A. Reinke, M. Tizabi, L. Ferrer, P. F. Jäger, and L. Maier-Hein. Deployment of Image Analysis Algorithms Under Prevalence Shifts. InMedical Image Computing and Computer
Assisted Intervention – MICCAI 2023: 26th International Conference, Vancouver, BC, Canada, October 8–12,
2023, Proceedings, Part III, pages 389–399, Berlin, Heidelberg, Oct. 2023. Springer-Verlag.
[16] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test.The Journal
of Machine Learning Research, 13(null):723–773, Mar. 2012.
[17] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick. Masked Autoencoders Are Scalable Vision Learners.
In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15979–15988,
New Orleans, LA, USA, June 2022. IEEE.
[18] K.He,X.Zhang,S.Ren,andJ.Sun. DeepResidualLearningforImageRecognition. In 2016IEEEConference
on Computer Vision and Pattern Recognition (CVPR), pages 770–778. IEEE, June 2016.
[19] S. Jang, S. Park, I. Lee, and O. Bastani. Sequential Covariate Shift Detection Using Classifier Two-Sample
Tests. InProceedings of the 39th International Conference on Machine Learning, pages 9845–9880. PMLR,
June 2022. ISSN: 2640-3498.
[20] J. J. Jeong, B. L. Vey, A. Bhimireddy, T. Kim, T. Santos, R. Correa, R. Dutt, M. Mosunjac, G. Oprea-Ilies,
G. Smith, M. Woo, C. R. McAdams, M. S. Newell, I. Banerjee, J. Gichoya, and H. Trivedi. The EMory BrEast
imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic
Mammographic Images. Radiology: Artificial Intelligence, 5(1):e220047, Jan. 2023. Publisher: Radiological
Society of North America.
[21] H.Kang,D.Luo,W.Feng,S.Zeng,T.Quan,J.Hu,andX.Liu.StainNet: AFastandRobustStainNormalization
Network. Frontiers in Medicine, 8, 2021.
[22] M. Karthik and D. Sohier. APTOS 2019 Blindness Detection Kaggle Challenge, 2019.
15

## Page 16 / 17

[23] L. M. Koch, C. F. Baumgartner, and P. Berens. Distribution shift detection for the postmarket surveillance of
medical AI algorithms: a retrospective simulation study.npj Digital Medicine, 7(1):1–11, May 2024. Publisher:
Nature Publishing Group.
[24] L. M. Koch, C. M. Schürch, C. F. Baumgartner, A. Gretton, and P. Berens. Deep Hypothesis Tests Detect
Clinically Relevant Subgroup Shifts in Medical Images, Mar. 2023. arXiv:2303.04862 [cs].
[25] F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland. Learning Deep Kernels for Non-Parametric
Two-Sample Tests. InProceedings of the 37th International Conference on Machine Learning, pages 6316–
6326. PMLR, Nov. 2020. ISSN: 2640-3498.
[26] D.Lopez-PazandM.Oquab. RevisitingClassifierTwo-SampleTests. In InternationalConferenceonLearning
Representations, 2017.
[27] W. Ma, C. Chen, S. Zheng, J. Qin, H. Zhang, and Q. Dou. Test-Time Adaptation with Calibration of Medical
Image Classification Nets for Label Distribution Shift. In L. Wang, Q. Dou, P. T. Fletcher, S. Speidel, and
S. Li, editors,Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, Lecture Notes
in Computer Science, pages 313–323, Cham, 2022. Springer Nature Switzerland.
[28] J. Merkow, A. Soin, J. Long, J. P. Cohen, S. Saligrama, C. Bridge, X. Yang, S. Kaiser, S. Borg, I. Tarapov,
and M. P. Lungren. CheXstray: A Real-Time Multi-Modal Monitoring Workflow for Medical Imaging AI. In
H. Greenspan, A. Madabhushi, P. Mousavi, S. Salcudean, J. Duncan, T. Syeda-Mahmood, and R. Taylor, editors, Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, pages 326–336, Cham,
2023. Springer Nature Switzerland.
[29] K. P. Murphy.Probabilistic machine learning: advanced topics. Adaptive computation and machine learning
series. The MIT Press, Cambridge, Massachusetts, 2023.
[30] S. Rabanser, S. Günnemann, and Z. Lipton. Failing Loudly: An Empirical Study of Methods for Detecting
Dataset Shift. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.,
2019.
[31] M. Roschewitz, G. Khara, J. Yearsley, N. Sharma, J. J. James, E. Ambrozay, A. Heroux, P. Kecskemethy,
T. Rijken, and B. Glocker. Automatic correction of performance drift under acquisition shift in medical image
classification. Nature Communications, 14(1):6608, Oct. 2023.
[32] M.Saerens,P.Latinne,andC.Decaestecker. AdjustingtheOutputsofaClassifiertoNewaPrioriProbabilities:
A Simple Procedure.Neural Computation, 14(1):21–41, Jan. 2002.
[33] B. Sahiner, W. Chen, R. K. Samala, and N. Petrick. Data drift in medical machine learning: implications and
potential remedies.British Journal of Radiology, 96(1150):20220878, Oct. 2023.
[34] L. Seyyed-Kalantari, H. Zhang, M. B. A. McDermott, I. Y. Chen, and M. Ghassemi. Underdiagnosis bias
of artificial intelligence algorithms applied to chest radiographs in under-served patient populations.Nature
Medicine, 27(12):2176–2182, Dec. 2021. Number: 12 Publisher: Nature Publishing Group.
[35] N.Sharma,A.Y.Ng,J.J.James,G.Khara,E.Ambrozay,C.C.Austin,G.Forrai,G.Fox,B.Glocker,A.Heindl,
E. Karpati, T. M. Rijken, V. Venkataraman, J. E. Yearsley, and P. D. Kecskemethy. Multi-vendor evaluation
of artificial intelligence as an independent reader for double reading in breast cancer screening on 275,900
mammograms. BMC Cancer, 23(1):460, May 2023.
16

## Page 17 / 17

[36] G.Shih,C.C.Wu,S.S.Halabi,M.D.Kohli,L.M.Prevedello,T.S.Cook,A.Sharma,J.K.Amorosa,V.Arteaga,
M.Galperin-Aizenberg, R.R.Gill, M.C.Godoy, S.Hobbs, J.Jeudy, A.Laroia, P.N.Shah, D.Vummidi, K.Yaddanapudi, and A. Stein. Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
Annotations of Possible Pneumonia.Radiology: Artificial Intelligence, 1(1):e180041, Jan. 2019. Publisher:
Radiological Society of North America.
[37] K.Stacke,G.Eilertsen,J.Unger,andC.Lundstrom. MeasuringDomainShiftforDeepLearninginHistopathology. IEEE journal of biomedical and health informatics, 25(2):325–336, Feb. 2021.
[38] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers. ChestX-ray8: Hospital-scale Chest Xray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax
Diseases. In2017IEEEConferenceonComputerVisionandPatternRecognition(CVPR) ,pages3462–3471,
July 2017.
[39] H.Wen,A.Betken,andH.Hang. ClassProbabilityMatchingwithCalibratedNetworksforLabelShiftAdaption.
In Proceedings of The Twelfth International Conference on Learning Representations, Apr. 2024.
[40] S. Xie, Z. Zheng, L. Chen, and C. Chen. Learning Semantic Representations for Unsupervised Domain
Adaptation. In Proceedings of the 35th International Conference on Machine Learning, pages 5423–5432.
PMLR, July 2018. ISSN: 2640-3498.
[41] W. Yan, L. Huang, L. Xia, S. Gu, F. Yan, Y. Wang, and Q. Tao. MRI Manufacturer Shift and Adaptation: IncreasingtheGeneralizabilityofDeepLearningSegmentationforMRImagesAcquiredwithDifferentScanners.
Radiology. Artificial Intelligence, 2(4):e190195, July 2020.
[42] Y. Yang, H. Zhang, J. W. Gichoya, D. Katabi, and M. Ghassemi. The limits of fair medical imaging AI in
real-world generalization.Nature Medicine, pages 1–11, June 2024.
[43] G. Zamzmi, K. Venkatesh, B. Nelson, S. Prathapan, P. Yi, B. Sahiner, and J. G. Delfino. Out-of-Distribution
Detection and Radiological Data Monitoring Using Statistical Process Control.Journal of Imaging Informatics
in Medicine, Sept. 2024.
[44] K.Zhou, Z.Liu,Y.Qiao,T.Xiang,andC.C.Loy. DomainGeneralization: ASurvey. IEEETransactionsonPattern Analysis and Machine Intelligence, 45(4):4396–4415, Apr. 2023. Conference Name: IEEE Transactions
on Pattern Analysis and Machine Intelligence.
[45] Y. Zhou, M. A. Chia, S. K. Wagner, M. S. Ayhan, D. J. Williamson, R. R. Struyven, T. Liu, M. Xu, M. G. Lozano,
P. Woodward-Court, Y. Kihara, A. Altmann, A. Y. Lee, E. J. Topol, A. K. Denniston, D. C. Alexander, and P. A.
Keane. A foundation model for generalizable disease detection from retinal images.Nature, 622(7981):156–
163, Oct. 2023. Publisher: Nature Publishing Group.
[46] L. Zuo, B. E. Dewey, Y. Liu, Y. He, S. D. Newsome, E. M. Mowry, S. M. Resnick, J. L. Prince, and A. Carass.
UnsupervisedMRharmonizationbylearningdisentangledrepresentationsusinginformationbottlenecktheory.
NeuroImage, 243:118569, Nov. 2021.
17
