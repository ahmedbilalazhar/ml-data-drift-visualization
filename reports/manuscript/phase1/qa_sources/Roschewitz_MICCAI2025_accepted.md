# Accepted MICCAI 2025 paper: verification copy

Source: https://papers.miccai.org/miccai-2025/paper/4612_paper.pdf
Retrieved: 2026-10-04. This supersedes the supplied arXiv v3 for final-paper results.

## Page 1

Automatic dataset shift identification to support
safe deployment of medical imaging AI
Mélanie Roschewitz, Raghav Mehta, Charles Jones, and Ben Glocker
Imperial College London
m.roschewitz21@imperial.ac.uk
Abstract.Shifts in data distribution can substantially harm the per-
formance of clinical AI models and lead to misdiagnosis. Hence, vari-
ous methods have been developed to detect the presence of such shifts
at deployment time. However, root causes of dataset shifts are varied,
and the choice of shift mitigation strategies highly depends on the pre-
cise type of shift encountered at test time. As such,detecting test-time
dataset shift is not sufficient: preciselyidentifying which type of shift
has occurred is critical. In this work, we propose the first unsupervised
dataset shift identification framework, effectively distinguishing between
prevalence shift, covariate shift and mixed shifts. We show the effective-
ness of the proposed shift identification framework across three differ-
ent imaging modalities (chest radiography, digital mammography, and
retinal fundus images) on five types of real-world dataset shifts, us-
ing five large publicly available datasets. Code is publicly available at
https://github.com/biomedia-mira/shift_identification.
Keywords: Dataset shift · Monitoring · Distribution shift
1 Introduction
Machine learning models are notoriously sensitive to changes in the input data
distribution, a phenomenon commonly referred to as dataset shift [28]. This is
particularly problematic in clinical settings, where dataset shift is a common
occurrence and may arise from various factors [3]. Changes in the frequency
of disease positives over time or across geographical regions causeprevalence
shift [9] (also known as label shift). The use of different acquisition protocols or
scanners [22,23], or a change in patient demographics [26] can induce shifts in
image characteristics, known ascovariate shift. We illustrate examples of real-
world shifts in Fig.1. Dataset shift can dramatically affect the performance of
AI, lead to clinical errors such as misdiagnosis and is recognised as the funda-
mental barrier hindering AI adoption [21,18,8,20]. It is hence crucial to imple-
ment safeguards allowing not only effective detection of the presence of shifts,
but importantly, reliable identification of the root causes. Comprehensive shift
detection and identification frameworks are key for the safe deployment and
continuous monitoring of AI in clinical practice.
Dataset shifts can be detected at deployment time by using statistical test-
ing to compare the distributions of incoming test data to the distribution of

## Page 2

2 M. Roschewitz et al.
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
Fig.1.Examples of dataset shifts in medical imaging. Reliably detecting and identi-
fying the nature of the shift is crucial to enable the safe deployment of ML systems.
the reference data (representative of the data used to validate the deployed AI
model). Significant progress has been made in this field where state-of-the-art
methods can detect various types of real-world shifts [14,17]. Shifts between test
and reference data can either be detected at the output level by comparing dis-
tributions of model outputs, or at the input level by comparing low-dimensional
feature representations of input images [17]. In this study, we also show that
different types of shifts require different shift detection approaches, and that
self-supervised (SSL) image encoders [4], yield excellent low-dimensional feature
representations for shift detection.
Whiledetectingdataset shifts is important, it is insufficient for the safe de-
ployment of AI. Besides knowing that there is a problem, we need to be able to
identifythe precise type of shift to take the necessary actions, implement pre-
ventive measures, and safeguard against harm caused by AI errors. Indeed, many
domain adaptation techniques are shift-specific: applying the wrong mitigation
technique may, in the best case, be ineffective in resolving the shift or, in the
worst case, severely harm model performance or calibration. For example, preva-
lence shifts can often be mitigated with lightweight output recalibration tech-
niques [1,24], but these rely on the assumption that no other types of shift are
present. Applying such label shift adaptation methods when the shift is actually
caused by covariate shift may drastically degrade model calibration and clinical
metrics. In contrast, covariate shifts require more advanced domain adaptation
techniques or model fine-tuning [29,12,25]. For example, image-harmonisation
techniques (e.g. [12]) or automatic correction methods (e.g. [18]) effectively mit-
igate effects of acquisition shifts on model performance but will fail in the case
of prevalence shift. The difficulty lies in the fact that a change in image charac-
teristics may cause similar changes in the distribution over model outputs as a
change in disease prevalence [18], and determining the cause of an observed shift
can be challenging. Despite its importance, automatic dataset shift identification
has remained an open problem.
In this work, we address this issue by proposing a dataset shiftidentification
frameworkcapableofidentifyingtherootcauseoftheunderlyingshift,effectively
separating (i) prevalence shift, (ii) covariate shift and (iii) mixed shift (both
prevalence and covariate shifts). To the best of our knowledge, this is the first

## Page 3

Automatic dataset shift identification 3
Estimate prevalence on test set.
30% positives
Resample reference set.
30% positivesTest set
Prevalence-adjusted reference
Significant differences 
PREVALENCE SHIFT ONLYNo differenceIf differences disappear after prevalence adjustmentPREVALENCE SHIFT PRESENT
Else COVARIATE SHIFT ONLY
B. SHIFT IDENTIFICATION
Reference set
Test set
EncoderTask model
Encoder
P(y) = 0.8
P(y) = 0.9
Test output distribution
Reference feature distribution
A. SHIFT DETECTIONNo differences
Significant differencesSHIFT DETECTED
PIPELINE OVERVIEW
Unlabelled test set 
Model decisions
No shift
Shift detected
A. SHIFT DETECTIONB. SHIFT IDENTIFICATION
NO SHIFT
PREVALENCE + COVARIATE SHIFT
COVARIATE SHIFTPREVALENCE SHIFT
Annotated reference set
Test feature distribution
Reference output distribution
COMPARE
HEALTHY
COMPARE
NO SHIFT1 2
345
COVARIATE SHIFT PRESENT
Original reference
Prev.-adjusted reference
Test
Test
COMPARE
COMPARE
6 PREVALENCE + COVARIATE SHIFT
Task model
Fig.2. Proposed dataset shift identification pipeline. Contrarily to previous
works, we do not simply detect thepresenceof shifts but alsoidentifythe nature of the
shift, leveraging both task model outputs and features from self-supervised encoders.
framework able to identify the type of test-time shifts in an unsupervised manner
for imaging data, beyond solely detecting shifts. An in-depth evaluation across
three different clinical applications (chest radiography, digital mammography,
and retinal fundus images) on five types of real-world dataset shifts demonstrates
that our framework accurately distinguishes between prevalence shifts, covariate
shifts, and mixed shifts across various scenarios.
2 Background
Shift definitions.Formally, letXdenote the input image andYthe target.La-
bel shift[16] (or prevalence shift) occurs when label distribution changes across
domains i.e.P ref (Y)̸=P test(Y), the conditional distributions are preserved, i.e.
Pref (X|Y) =P test(X|Y), whereP ref andP test denote distributions on refer-
ence and target domains respectively. Conversely,covariate shiftoccurs when
Pref (X)̸=P test(X), while conditional distributions are preserved [16]. Acquisi-
tion and subpopulation shifts are cases of covariate shifts as they directly affect
image appearance.
Dataset shift detection.Several paradigms have been proposed for dataset shift
detection. The simplest method to implement consists of comparing distributions
of a classifier’s outputs between the reference and test domain, proposed by

## Page 4

4 M. Roschewitz et al.
Rabanser et al. [17] and referred to as ‘Black Box Shift Detection’ (BBSD).
Specifically, softmax model outputs are collected for all samples in the reference
andtestsets.Then,foreachclass,aseparateunivariateKolmogorov-Smirnov(K-
S) test is run to determine if the class-wise predicted probabilities distributions
differ between reference and test domain, the overall significance of the shift is
then determined after applying Bonferroni [7] correction for multiple testing.
Rabanser et al. [17] also propose another type of shift detector where reference
and test data input distribution are compared using a feature-based approach.
In this test, the input sample (image) first gets projected to a smaller dimension,
e.g. through a pretrained neural network encoder and the shift is then measured
using the ‘Maximum Mean Discrepancy’ (MMD) permutation test. Other shift
detection approaches consist of training a domain classifier to classify samples
between the reference and test domains [15,10] and using the accuracy of this
classifierasaproxyformeasuringdistancesbetweendistributions.Onedrawback
of this approach is its high computational cost: for every test set, a new domain
classifier must be trained, which is highly impractical in continuous monitoring
scenarios.
3 Methods
3.1 Dataset shift identification pipeline
We propose a framework for identifying whether dataset shift is caused by preva-
lence shift, by covariate shift or by a mix of both. The approach is divided in two
stages (Fig. 2). First, we perform standard dataset shift detection to separate
the ‘shift’ from the ‘no shift’ cases (Fig. 2, A). For this step, we use a dual de-
tection approach, combining signals from task model outputs and features from
self-supervised (SSL) encoders to detect shifts. In details, we independently run
the BBSD and the MMD shift detection tests; this yieldsCp-values for the
BBSD test (one per class), and one p-value for the MMD permutation test,
we then apply Bonferroni correction on theC+ 1p-values to get the overall
significance. If a shift is detected we then proceed to shiftidentification. This
starts with estimating the prevalence in the test set (Fig. 2, B.3). For this, we
can leverage the prevalence shift adaptation literature, where various methods
have been proposed to estimate the density ratioP ref (Y)/P test(Y)[1,19,24].
Here, we use the state-of-the-art ‘class probability matching with calibrated net-
works’ (CPMCN) method [24] to estimate this ratio and the test set prevalence.
Next, we resample the reference set to match this estimated prevalence (Fig. 2,
B.4). We then first compare feature distributions between prevalence-adjusted
reference and test set (Fig. 2, B.5): if differences are no longer significant after
adjusting the prevalence, the shift is attributed to prevalence shift. Conversely,
if differences persist after adjusting the prevalence, then covariate shift is neces-
sarily present. In this case, we compare model output distributions with BBSD
to determine whether prevalence shift is also present (Fig. 2, B.6). Precisely, if
there were significant differences in model output distributions before adjusting
the prevalence, but this shift disappears after adjusting the prevalence, we know

## Page 5

Automatic dataset shift identification 5
that prevalence shift is also responsible for the observed shift, in this case we
conclude that the observed shift is a case of mixed shift (prevalence + covariate
shift). Else, we conclude that the shift is attributed to covariate shift only.
3.2 Datasets and shift generation
We evaluate our methods on four different datasets covering three different imag-
ing modalities: (i) chest radiography using PadChest [2] (collected in Spain, with
two scanners); (ii) mammography using the EMBED [11] dataset (mammograms
acquired in the US with six different scanners), and (iii) fundus images for which
we create ‘RETINA’ a multi-domain dataset by combining three different public
datasets: the Kaggle Diabetic Retinopathy Detection dataset [6], the Kaggle Ap-
tos Blindness Detection dataset [13] and the Messidor-v2 dataset [5] (covering
three countries US, India and France and various acquisition devices from high
quality scanners to phone pictures). To study prevalence shift detection, we as-
sociate each dataset with a downstream task. For chest radiography datasets we
focus on pneumonia detection, for mammography on breast density assessment
(4 classes), and for retinal images on binary diabetic retinopathy classification.
Then, we study various types of covariate shifts. For PadChest, we study gender
shift by varying the proportion of female patients in the test set. Moreover, Pad-
Chest contains scans acquired with two scanners, ‘Phillips’ (40%) and ‘Imaging’
(60%). This allows to simulate different levels of acquisition shift, by varying
the proportion of Phillips scans in the test set. Similarly, for EMBED [11], we
study acquisition shift by varying the distribution of scanners in the test set.
This dataset offers a complementary view to PadChest, with a multi-class task
of interest and providing even more flexibility for simulating diverse acquisition
shifts (six scanners). Note that in EMBED each exam comprises 4 mammograms
(left/right breasts and MLO/CC views). We excluded all exams that did not con-
tain exactly four images and kept exactly one exam per patient, and ensure that
test set sampling was done at the exam level. Finally, for the RETINA dataset,
we simulate covariate shifts by varying the proportion of samples coming from
each underlying dataset (Aptos [13], Kaggle DR [6] and Messidor [5]).
Implementation details
4 Results
4.1 Different shifts require different shift detectors
Prior to diving into shiftidentification, we first investigate which types of shifts
are successfullydetectedby prominent dataset shift detection methods. We com-
pare two families of shift detectors: model output-based (BBSD) and feature-
based detectors (MMD). We additionally test a dual approach that combines
both approaches for improved shift detection (‘Duo’). For feature-based shift
detection, any pretrained network can be used as feature extractor, we could

## Page 6

6 M. Roschewitz et al.
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* *
* *
* *
Prevalence: 20% (+16%)
PadChest  - Prevalence shift
Original prevalence: 4%
50 100 250
Number of exams (each with 4 images)
0
25
50
75
100Detected shift (%)
* *
* * * *
Density class distribution: 15%, 35%, 35%, 15%
EMBED  - Prevalence shift
Original density distribution: 7%, 37%, 47%, 7%
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
PadChest  - Gender shift
Original proportion of females: 51%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * * * *
*
* *
Proportion of Phillips: 25% (-17%)
PadChest - Acquisition shift
Original proportion of Phillips: 42%
50 100 250
Number of exams (each with 4 images)
0
25
50
75
100Detected shift (%)
* * * *
* * * * *
Scanner distribution: 55%,0%,10%,10%,15%,10%
EMBED - Acquisition shift
Original scanner distribution: 79%,0%,5%,4%,7%,5%
100 250 1000
Number of test images
0
25
50
75
100Detected shift (%)
* * * * * * * * *
Domain distribution [0.05, 0.3, 0.65]
RETINA  - Acquisition shift
 Original domain distribution: [0.04, 0.09, 0.87]
Shift detector
BBSD test (task model outputs)
MMD (Supervised ImageNet features)
MMD (Task model features)
MMD (SimCLR ImageNet features)
MMD (SimCLR Modality Specific features) Duo BBSD + MMD (SimCLR ImageNet)
Shift detector
BBSD test (task model outputs)
MMD (Supervised ImageNet features)
MMD (Task model features)
MMD (SimCLR ImageNet features)
MMD (SimCLR Modality Specific features) Duo BBSD + MMD (SimCLR ImageNet)
Fig.3. Shift detectors comparison. We studied prevalence shift as well as two sub-
types of covariate shifts: subpopulation and acquisition shift. Shift detection rate is
computed over 200 bootstrap samples. Feature-based detection (MMD) is best for co-
variate shift detection and model-output based detection (BBSD) is best for prevalence
shift. For each test set size, the best detector along with all detectors not significantly
different from it, are denoted with∗(Fisher’s exact test, at level .05, with Bonferroni).
for example use the task model directly. However, this may not be the best
choice as learned features will be heavily skewed towards encoding character-
istics specifically relevant to that task as opposed to encoding more generic
image representations [27]. Hence, we here explore the potential of SSL image
encoders for shift detection as these encoders learn to effectively summarise the
information encoded in a given image. We compare the performance of MMD de-
tection for four different encoders: (i) ‘Supervised ImageNet’ trained to perform
classification on ImageNet; (ii) ‘Task model’ trained to perform the downstream
classification task; (iii) ‘SSL ImageNet’ trained on ImageNet data only; (iv) ‘SSL
Modality Specific’ trained on the same modality as the reference dataset.
Fig. 3 show the shift detection rates for various dataset-shift combination
across shift detectors. We can see that for prevalence shifts, output-based shift
detection performs significantly better than all feature-based tests. This is in-
tuitive as a change in prevalence should directly be reflected by a change in
the distribution of task model outputs. The exact opposite is true for covari-
ate shifts, where most feature-based detectors perform substantially better than
output-based detectors, regardless of whether we look at acquisition or subpop-
ulation shifts. The results also show that for optimal detection of more subtle
covariateshifts,itiscrucialtouseanSSLencoder,asSSLencodersoffersubstan-
tially better detection rates than their supervised counterparts. The SSL model
trained on ImageNet data was particularly effective and, in some cases even bet-
ter than the modality-specific SSL model. Given the orthogonal behaviour of
output-based and feature-based tests, we introduce a dual detection approach

## Page 7

Automatic dataset shift identification 7
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Prevalence:
20% (+16%)
100 250 500 1000
Number of test images
0
25
50
75
100
Prevalence:
30% (+26%)
PadChest - Prevalence shift
Original prevalence: 4%
50 100 250
Number of exams
0
25
50
75
100Identified shift (%)
Prevalence:
15%,35%,35%,15%
50 100 250
Number of exams
0
25
50
75
100
Prevalence:
10%,20%,60%,10%
EMBED - Prevalence shift
Original class distribution: 7%,37%,47%,7%
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Prevalence:
65% (-13%)
100 250 500 1000
Number of test images
0
25
50
75
100
Prevalence:
100% (+22%)
RETINA - Prevalence shift
Original prevalence: 78%
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Proportion of Phillips scans:
25% (-17%)
100 250 500 1000
Number of test images
0
25
50
75
100
Proportion of Phillips scans:
75% (+33%)
PadChest - Acquisition shift
Original proportion of Phillips scans: 42%
50 100 250
Number of exams
0
25
50
75
100Identified shift (%)
Scanner distribution:
55%,0%,10%,10%,15%,10%
50 100 250
Number of exams
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
Number of test images
0
25
50
75
100Identified shift (%)
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
RETINA - Acquisition shift
Original site distribution: 4%,9%,87%
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Females: 25% (-26%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 75% (+24%)
PadChest - Gender shift
Original proportion of females: 51%
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Females: 25% (-26%)
Disease prev.: 15% (+11%)
100 250 500 1000
Number of test images
0
25
50
75
100
Females: 100% (+49%)
Disease prev.: 15% (+11%)
PadChest - Gender + Prevalence shift
Original: females 51%, disease prevalence 4%
100 250 500 1000
Number of test images
0
25
50
75
100Identified shift (%)
Proportion of Phillips scans:
75% (+33%)
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
75% (+33%)
Disease prevalence:
25% (+21%)
PadChest - Acquisition + Prevalence shift
Original proportion of Phillips scans: 42%
Original disease prevalence: 4%
50 100 250
Number of exams
0
25
50
75
100Identified shift (%)
Scanner distribution:
55%,0%,10%,10%,15%,10%
Density distribution:
0%, 50%, 50%, 0%
50 100 250
Number of exams
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
Number of test images
0
25
50
75
100Identified shift (%)
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
RETINA - Acquisition + Prevalence shift
Original site distribution: 4%, 9%, 87%
Original disease prevalence: 78%
Prevalence shift Covariate only Covariate + PrevalencePrevalence shift Covariate only Covariate + Prevalence Correct IncorrectPrevalence shift Covariate only Covariate + Prevalence Correct Incorrect
Fig.4. Shift identification accuracy. Across all datasets, the shift identification
framework is able to successfully detect and identify all types of shifts high accuracy.
combining responses from output-based and SSL feature-based shift detectors
(see Section 3). Results in Fig. 3 confirm that this ‘Duo’ approach performs
best overall across shifts and datasets. For example, in PadChest, with 1000 test
cases: MMD (ImageNet-SSL features) has 65% detection rate of prevalence shift,
100% for gender and acquisition shift, i.e. an average of 88% detection across
shifts, BBSD has an average of 60% across shifts, and Duo has an average of
detection rate close to 100%. We employ this ‘Duo’ approach for the detection
step of our shift identification pipeline.
4.2 Shift identification accuracy
In this work, we propose a framework able to precisely identify the type of shift
present in the test dataset (see Section 3). Evaluation results in Fig. 4 show that
our framework is capable of distinguishing between prevalence shifts, covariate
shifts and mixed shifts with high accuracy across all datasets and types of shifts.
For prevalence shifts, the average shift identification rate, across datasets and
shift levels, is91%with 500 test images (top row). Similarly, for covariate shifts

## Page 8

8 M. Roschewitz et al.
(acquisition and gender shifts), when using a test set size of 500 images (250
exams on EMBED), the identification accuracy is greater than 80% for any shift
level and dataset and the average accuracy is 87%. Importantly, the framework
is also able to accurately distinguish between cases of covariate shift only and
cases of covariate and prevalence shifts. With a test set of 500 images (resp. 250
exams on EMBED), mixed gender and prevalence shifts are correctly identified
as mixed shifts with an average accuracy of75%across all tested scenarios.
Overall, more subtle shifts are best detected with larger test sets whereas larger
shifts can be detected with smaller test sets.
5 Discussion
Shift identification is critical for monitoring AI systems in deployment and for
the root cause analysis of AI performance drift. By analysing common dataset
shift detection paradigms, we find that different types of shifts require different
types of shift detectors. Our analysis also demonstrates that encoders trained
in a self-supervised manner yield features with substantially higher shift detec-
tion power than supervised counterparts. Maybe surprisingly, our results show
that generic encoders trained in a self-supervised manner on ImageNet provide
highly discriminative features for medical image dataset shift detection, trans-
portable across datasets. Following these findings, we propose a lightweight and
practical shift identification framework, capable of detecting and identifying im-
portant real-world dataset shifts, showing high accuracy in correctly identifying
the type of shift present in the test set across all modalities, tasks and various
levels of shift intensity. Our results demonstrate the practical value of the shift
identification method, which does not require any training at test time, nor any
ground truth labels or annotations on the test domain data. The use of a readily
available SSL encoder trained on ImageNet data for feature extraction dispenses
us from training any additional model for shift detection and identification pur-
poses. Combining signals from both feature-based and model output-based shift
detectors yields reliable and consistent detection and identification across all
types of shifts. Importantly, our method not only separates covariate shifts from
prevalence shifts but also reliably detects when both types of shifts are present
in the test set, rendering the proposed method applicable to many real-world
deployment scenarios.
In terms of limitations, we note that in the case of covariate shift, on its
own, the proposed framework does not allow for a more fine-grained identifica-
tion of the origin of shift, e.g. the distinction between population and acquisition
shifts. To allow for an even more precise sub-type shift identification, integrating
metadata statistics in the pipeline could complement the framework. However,
relying solely on metadata monitoring is insufficient as it only enables the detec-
tion of shifts affecting the specific attributes collected at test time. The proposed
framework can help uncover shifts that are not detectable by means of simply
tracking population metadata. In this context, our shift identification framework
offers an important safeguard for deploying AI models in clinical practice.

## Page 9

Automatic dataset shift identification 9
Acknowledgments
M.R.isfundedbyanImperialCollegeLondonPresident’sPhDScholarshipanda
Google PhD Fellowship. R.M. is funded through the European Union’s Horizon
Europe research and innovation programme under grant agreement 10108030.
C.J. is supported by Microsoft Research and EPSRC through the Microsoft PhD
Scholarship Programme. B.G. acknowledges support from the Royal Academy
of Engineering as part of his Kheiron Medical Technologies/RAEng Research
Chair in Safe Deployment of Medical Imaging AI.
Disclosure of interests
B.G. is part-time employee of DeepHealth. No other competing interests.
References
1. Alexandari, A., Kundaje, A., Shrikumar, A.: Maximum Likelihood with Bias-
Corrected Calibration is Hard-To-Beat at Label Shift Adaptation. In: Proceedings
of the 37th International Conference on Machine Learning. pp. 222–232. PMLR
(Nov 2020), iSSN: 2640-3498
2. Bustos, A., Pertusa, A., Salinas, J.M., de la Iglesia-Vayá, M.: PadChest: A large
chest x-ray image dataset with multi-label annotated reports. Medical Image Anal-
ysis66, 101797 (Dec 2020)
3. Castro, D.C., Walker, I., Glocker, B.: Causality matters in medical imaging. Nature
Communications11(1), 3673 (Jul 2020)
4. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A Simple Framework for Con-
trastive Learning of Visual Representations. In: Proceedings of the 37th Interna-
tional Conference on Machine Learning. pp. 1597–1607. PMLR (Nov 2020)
5. Decencière, E., Zhang, X., Cazuguel, G., Lay, B., Cochener, B., Trone, C., Gain,
P., Ordonez, R., Massin, P., Erginay, A., Charton, B., Klein, J.C.: Feedback on a
publicly distributed image database: the messidor database. Image Analysis and
Stereology33(3), 231–234 (Aug 2014), number: 3
6. Dugas, E., Jared, J., Cukierski, W.: Diabetic Retinopathy Detection Kaggle Chal-
lenge (2015), https://kaggle.com/competitions/diabetic-retinopathy-detection
7. Dunn, O.J.: Multiple Comparisons among Means. Journal of the American Statis-
tical Association56(293), 52–64 (Mar 1961)
8. Finlayson Samuel G., Subbaswamy Adarsh, Singh Karandeep, Bowers John, Kupke
Annabel, Zittrain Jonathan, Kohane Isaac S., Saria Suchi: The Clinician and
Dataset Shift in Artificial Intelligence. New England Journal of Medicine385(3),
283–286 (Jul 2021)
9. Godau, P., Kalinowski, P., Christodoulou, E., Reinke, A., Tizabi, M., Ferrer, L.,
Jäger, P.F., Maier-Hein, L.: Deployment of Image Analysis Algorithms Under
Prevalence Shifts. In: MICCAI 2023. pp. 389–399. Springer (Oct 2023)
10. Jang, S., Park, S., Lee, I., Bastani, O.: Sequential Covariate Shift Detection Using
Classifier Two-Sample Tests. In: Proceedings of the 39th International Conference
on Machine Learning. pp. 9845–9880. PMLR (Jun 2022), iSSN: 2640-3498

## Page 10

10 M. Roschewitz et al.
11. Jeong, J.J., Vey, B.L., Bhimireddy, A., Kim, T., Santos, T., Correa, R., Dutt,
R., Mosunjac, M., Oprea-Ilies, G., Smith, G., Woo, M., McAdams, C.R., Newell,
M.S., Banerjee, I., Gichoya, J., Trivedi, H.: The EMory BrEast imaging Dataset
(EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and
Diagnostic Mammographic Images. Radiology: Artificial Intelligence5(1), e220047
(Jan 2023). https://doi.org/10.1148/ryai.220047
12. Kang, H., Luo, D., Feng, W., Zeng, S., Quan, T., Hu, J., Liu, X.: StainNet: A Fast
and Robust Stain Normalization Network. Frontiers in Medicine8(2021)
13. Karthik, M., Sohier, D.: APTOS 2019 Blindness Detection Kaggle Challenge
(2019), https://kaggle.com/competitions/aptos2019-blindness-detection
14. Koch, L.M., Baumgartner, C.F., Berens, P.: Distribution shift detection for the
postmarket surveillance of medical AI algorithms: a retrospective simulation study.
npj Digital Medicine7(1), 1–11 (May 2024)
15. Lopez-Paz, D., Oquab, M.: Revisiting Classifier Two-Sample Tests. In: ICLR (Jul
2017)
16. Murphy,K.P.:ProbabilisticMachineLearning:AdvancedTopics.MITPress(2023)
17. Rabanser, S., Günnemann, S., Lipton, Z.: Failing Loudly: An Empirical Study of
Methods for Detecting Dataset Shift. In: Advances in Neural Information Process-
ing Systems. vol. 32. Curran Associates, Inc. (2019)
18. Roschewitz, M., Khara, G., Yearsley, J., Sharma, N., James, J.J., Ambrozay, E.,
Heroux, A., Kecskemethy, P., Rijken, T., Glocker, B.: Automatic correction of
performance drift under acquisition shift in medical image classification. Nature
Communications14(1), 6608 (Oct 2023)
19. Saerens, M., Latinne, P., Decaestecker, C.: Adjusting the Outputs of a Classifier
to New a Priori Probabilities: A Simple Procedure. Neural Computation14(1),
21–41 (Jan 2002)
20. Sahiner, B., Chen, W., Samala, R.K., Petrick, N.: Data drift in medical ma-
chine learning: implications and potential remedies. British Journal of Radiology
96(1150), 20220878 (Oct 2023)
21. Seyyed-Kalantari, L., Zhang, H., McDermott, M.B.A., Chen, I.Y., Ghassemi, M.:
Underdiagnosis bias of artificial intelligence algorithms applied to chest radio-
graphs in under-served patient populations. Nature Medicine27(12), 2176–2182
(Dec 2021)
22. Sharma, N., Ng, A.Y., James, J.J., Khara, G., Ambrozay, E., Austin, C.C., Forrai,
G., Fox, G., Glocker, B., Heindl, A., Karpati, E., Rijken, T.M., Venkataraman, V.,
Yearsley, J.E., Kecskemethy, P.D.: Multi-vendor evaluation of artificial intelligence
as an independent reader for double reading in breast cancer screening on 275,900
mammograms. BMC Cancer23(1) (May 2023)
23. Stacke, K., Eilertsen, G., Unger, J., Lundstrom, C.: Measuring Domain Shift for
Deep Learning in Histopathology. IEEE journal of biomedical and health infor-
matics25(2), 325–336 (Feb 2021)
24. Wen, H., Betken, A., Hang, H.: Class Probability Matching with Calibrated Net-
works for Label Shift Adaption. In: Proceedings of The Twelfth International Con-
ference on Learning Representations (Apr 2024)
25. Xie, S., Zheng, Z., Chen, L., Chen, C.: Learning Semantic Representations for
Unsupervised Domain Adaptation. In: Proceedings of the 35th International Con-
ference on Machine Learning. pp. 5423–5432. PMLR (Jul 2018)
26. Yang, Y., Zhang, H., Gichoya, J.W., Katabi, D., Ghassemi, M.: The limits of fair
medical imaging AI in real-world generalization. Nature Medicine pp. 1–11 (Jun
2024)

## Page 11

Automatic dataset shift identification 11
27. Zamzmi, G., Venkatesh, K., Nelson, B., Prathapan, S., Yi, P., Sahiner, B., Delfino,
J.G.: Out-of-Distribution Detection and Radiological Data Monitoring Using Sta-
tistical Process Control. Journal of Imaging Informatics in Medicine (Sep 2024)
28. Zhou,K.,Liu,Z.,Qiao,Y.,Xiang,T.,Loy,C.C.:DomainGeneralization:ASurvey.
IEEE Transactions on Pattern Analysis and Machine Intelligence45(4), 4396–4415
(Apr 2023)
29. Zuo, L., Dewey, B.E., Liu, Y., He, Y., Newsome, S.D., Mowry, E.M., Resnick, S.M.,
Prince, J.L., Carass, A.: Unsupervised MR harmonization by learning disentangled
representationsusinginformationbottlenecktheory.NeuroImage243,118569(Nov
2021)

