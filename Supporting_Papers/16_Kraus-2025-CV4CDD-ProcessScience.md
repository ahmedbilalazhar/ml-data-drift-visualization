# Machine learning-based detection of concept drift in business processes (CV4CDD-4D)

**Authors:** A. Kraus & H. van der Aa

**Venue:** Process Science (2025) — New venue, do NOT count as Q1 without evidence

*Source PDF: `16_Kraus-2025-CV4CDD-ProcessScience.pdf`*

*Converted to markdown following the Selected_Papers strategy (full-text extraction, page-ordered). Verify title/authors/year against publisher before citing.*

---

## Page 1 / 26

Open Access
© The Author(s) 2025. Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits 
use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original 
author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third 
party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or 
exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://
creativecommons.org/licenses/by/4.0/.
RESEARCH
Kraus and van der Aa Process Science (2025) 2:5 
https://doi.org/10.1007/s44311-025-00012-w
Process Science
Machine learning-based detection 
of concept drift in business processes
Alexander Kraus1* and Han van der Aa2 
Abstract 
Concept drift in process mining occurs when a single event log includes data 
from multiple versions of a process, making the detection of such drifts essential 
for ensuring reliable process mining results. Although many techniques have been 
proposed, they exhibit limitations in accuracy and scope. Specifically, their accuracy 
diminishes when facing noise, varying drift types, or different levels of change severity. 
Additionally, these techniques primarily focus on detecting sudden and gradual drifts, 
overlooking the automated detection of incremental and recurring drifts. To address 
these limitations, we present CV4CDD-4D, a novel approach for automated concept 
drift detection that can identify sudden, gradual, incremental, and recurring drifts. Our 
approach follows an entirely different paradigm. Specifically, it employs a supervised 
machine learning model fine-tuned on a large collection of event logs with known 
concept drifts, enabling the model to learn how drifts manifest in event logs. The 
possibility to train such a model has recently emerged through a tool that generates 
event logs with known concept drifts. However, applying supervised machine learning 
remains challenging due to the complexities of encoding. To address this, we propose 
converting an event log into an image-based representation that captures process 
evolution over time, enabling the use of a state-of-the-art computer vision model 
to detect drifts. Our experiments show that our approach, compared to existing solutions, improves the accuracy and robustness to noise of drift detection while extending coverage to a broader range of drift types, highlighting the potential of this new 
paradigm.
Keywords: Process mining, Concept drift detection, Machine learning, Deep learning, 
Computer vision, Object detection
Introduction
Organizations execute various business processes to achieve their business objectives. 
These business processes are often supported by information systems that record data 
generated during the execution of process instances. Event logs represent snapshots of 
such recorded data over a specific period of time, forming the basis for process mining, 
a collection of techniques that examine how business processes are truly executed (van 
der Aalst,  2016). However, due to the dynamic nature of business environments, pro -
cesses under analysis are often not in a steady state but are rather subject to changes 
(Bose et  al. 2013). These changes can result in the presence of concept drifts in event 
*Correspondence: 
alexander.kraus@uni-mannheim.
de
1 Data and Web Science Group, 
University of Mannheim, 
Mannheim, Germany
2 Faculty of Computer Science, 
University of Vienna, Vienna, 
Austria

## Page 2 / 26

Page 2 of 26Kraus and van der Aa Process Science (2025) 2:5 
logs, i.e., situations in which an event log contains data from multiple versions of the 
same process. Concept drift detection aims to identify these different versions in order to 
prevent the contamination of process mining results that can occur from mixing them 
(Bose et al. 2013).
The importance of concept drift detection in process mining has been widely acknowledged (Van Der Aalst et al. 2011), resulting in a range of proposed concept drift detec -
tion techniques (Sato et  al. 2021). However, these existing techniques exhibit notable 
limitations concerning their accuracy and scope. With respect to their accuracy, the performance of existing techniques tends to significantly decline in situations where noise, 
varying types of drifts, or different levels of change severity are present in an event log. 
Such declines occur because the techniques depend on algorithmic design choices, often 
based on heuristic strategies and assumptions about how drifts manifest in event logs. 
Since, these assumptions are not applicable to all situations, algorithms based on them 
can be subject to issues such as a lack of robustness to noise or generally poor accuracy. 
With respect to their scope, existing techniques typically detect only a subset of the commonly-known drift types, typically just focusing on sudden drifts, with few techniques 
also considering gradual ones (Elkhawaga et al. 2020). The detection of more complex 
drifts, such as incremental and recurring drifts (Bose et al. 2013), is generally overlooked 
and has not yet been tackled in an automated manner. Due to these limitations, existing 
concept drift detection techniques are thus unable to provide a precise and comprehen -
sive understanding of how processes evolve over time.
To overcome this, we propose CV4CDD-4D, a concept drift detection approach that 
can detect sudden, gradual, incremental, and recurring drifts in an automated manner, 
with high accuracy. We achieve this by following an entirely different paradigm for drift 
detection in process mining. Specifically, unlike existing techniques, our approach uses 
a machine learning model trained on a large dataset of labeled event logs, enabling it 
to learn how drifts of different types manifest themselves in event logs. The possibil -
ity of training such a model has only recently emerged, thanks to a tool for generating 
large collections of logs with known concept drifts (Grimm et al. 2022). However, even 
with such data, the challenge of applying (supervised) machine learning to concept drift 
detection is far from straightforward. This difficulty stems from the challenge of captur -
ing the progression of an entire process over time, in a way that it can serve as suitable 
for input into a machine learning model. To address this challenge, we draw inspiration 
from research that uses image-based representations to encode multi-faceted data about 
processes (Pasquadibisceglie et al. 2020; Pfeiffer et al. 2021). Therefore, we first turn an 
event log into an image that visualizes differences in process behavior over time. This 
enables us to employ a state-of-the-art object detection model (Liu et al. 2020) (from the 
field of computer vision), fine-tuned on a large collection of event logs with known con -
cept drifts, to recognize where drifts occur in unseen event logs. Our experiments reveal 
the efficacy of this idea, showing that our approach significantly improves the state of the 
art in terms of accuracy, robustness to noise, and automation in detecting drifts, while 
covering a broader range of drift types.
This paper presents an extended and revised version of our earlier work (Kraus and 
van der Aa, 2024) in which we introduced the first version of our computer vision-based 
approach for concept drift detection in process mining. The current paper extends our

## Page 3 / 26

Page 3 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
earlier work in two main directions. First, we broadened the scope of our approach by 
including the detection of complex drift types, i.e., incremental and recurring drifts, in 
addition to the sudden and gradual drifts already covered by the original version. Sec -
ond, we considerably extended the evaluation of our approach with additional experi -
ments, which allowed us to investigate the performance of our approach in more depth. 
Specifically, beyond extending the existing experiments to the broader scope of our 
extended approach, we included a sensitivity analysis in which we investigate the impact 
of the user parameters on our approach’s performance, and performed a qualitative analysis and benchmark comparison on various real-world event logs.
In the remainder, we define the boundaries of the work in the “Problem scope” sec -
tion. Our CV4CDD-4D approach, including the input to the problem and desired output, 
is detailed in the “ Approach” section. The “Evaluation” section examines our approach 
against state-of-the-art techniques. Finally, the “Related work ” section reflects on prior 
research relevant to this work before the “Conclusion” section summarizes the paper.
Problem scope
Our work addresses the problem of detecting concept drifts in the control-flow perspective of a process based on data recorded in an event log. Conventionally, such concept 
drifts encompass four types: sudden, gradual, incremental, and recurring drifts (Bose 
et al. 2013), as illustrated in Fig. 1. We pose that these four types can be sub-divided into 
two groups:
Simple drifts. We jointly refer to sudden and gradual drifts as simple drifts, since they 
correspond to a single change in a process from one version to another:
• Sudden drift: A sudden drift describes an abrupt replacement of one process ver -
sion by another. For instance, Fig.  1 shows the replacement of process version v1 by 
version v2 at a certain moment, i.e., the change point p1 . This means that all process 
instances that start after p1 will follow process version v2 . Sudden drifts are often 
observed in scenarios such as emergency response planning, where airlines and air -
Fig. 1 Problem scope: detection of sudden, gradual, incremental and recurring drifts

## Page 4 / 26

Page 4 of 26Kraus and van der Aa Process Science (2025) 2:5 
ports may alter their security procedures in response to new regulations (Bose et al. 
2013).
• Gradual drift: A gradual drift describes a situation where the replacement of process 
version v1 by v2 involves a transition period in which both versions coexist. In these 
cases, after an initial change point p1 an increasing fraction of new process instances 
will follow version v2 , until the roll-out of that version is completed at change point 
p2 . Gradual drifts, for instance, will occur when an organization starts training its 
employees in a spread-out manner regarding a new way of working, so that more and 
more employees start following the new version during the training period.
Complex drifts. We jointly refer to incremental and gradual drifts as complex drifts, 
since they consist of several, related simple drifts:
• Incremental drift: An incremental drift occurs when one process version is replaced 
by another through a sequence of simple drifts, rather than at once. For instance, 
in Fig.  1, process version v1.0 is replaced by v2.0 through a sequence of sudden and 
gradual drifts. This results in two intermediate versions, v1.1 and v1.2 , along with a 
total of four change points. Generally, the simple drifts that make up an incremental 
drift must relate to the same transformation initiative and are characterized by rela -
tively low change severity, which measures the extent to which process behavior is 
impacted by a change. A notable example of this type of drift are the process changes 
that arise from agile business process management methodologies (Bose et al. 2013).
• Recurring drift: A recurring drift is characterized by a situation when different ver -
sions of a process reoccur in an alternating fashion. For instance, Fig.  1 illustrates a 
situation where a process switches between versions v1 and v2 , following a sequence 
of sudden and gradual drifts. Recurring drifts are, for instance, common in processes 
with seasonal patterns.
In the following section, we consider the relevant literature within the scope of the 
discussed problem.
Related work
Since establishing the problem and importance of concept drift detection in process 
mining more than a decade ago (Van Der Aalst et al. 2011), various techniques have been 
proposed to address this problem, as highlighted in recent literature reviews (Elkhawaga 
et al. 2020; Sato et al. 2021). We first discuss existing techniques for offline concept drift 
detection, since these tackle the same or a subset of the task that our work addresses. 
Afterwards, we also consider works that focus on others types of concept drift detection, 
such as in online or multi-perspective settings.
Offline concept drift detection. Table  1 provides an overview of existing concept drift 
detection techniques, including their scope and automation level.
Change point detection. The majority of existing techniques (first row) just focus on 
detecting change points in an event log since this is a required first step in concept drift 
detection (Seeliger et al. 2017; Zheng et al. 2017; Brockhoff et al. 2020; Lin et al. 2020;

## Page 5 / 26

Page 5 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
Nguyen et al. 2018; Hompes et al. 2015; Neto et al. 2023; Meira Neto et al. 2024; Sato 
et al. 2021; Lu et al. 2021; Maaradji et al. 2015; Gallego-Fontenla et al. 2021; Nguyen et al. 
2018; Luengo and Sepúlveda, 2012; Impedovo et al. 2020; Weber et al. 2011; Accorsi and 
Stocker 2012; Manoj Kumar et  al. 2015; Xu et  al. 2023; Yang et  al. 2021). They use a 
wide range of ways to tackle this task, including statistical testing, various kinds of fea -
ture representations, windowing strategies, change and trend detection, and clustering. 
Many of the techniques achieve good results, as demonstrated in a recent evaluation 
framework (Adams et  al. 2023). However, our evaluation experiments show that our 
proposed CV4CDD-4D approach outperforms them in terms of accuracy and robustness 
to noise when detecting change points. Furthermore, these techniques do not go beyond 
the detection of change points, meaning that they are only able to recognize when a pro -
cess’s behavior significantly changed, but they do not provide insights into what type 
of drift actually occurred 1. As a result, they do not provide information about how the 
process evolved over time.
Drift type detection. A few techniques (Bose et al. 2013; Martjushev et al. 2015; Maar -
adji et al. 2017; Yeshchenko et al. 2021) go beyond change point detection and are also 
able to detect drifts and their types (though often limited in scope and automation).
Bose et al. (2013) introduced concept drift detection in process mining, presenting a 
method for automatically detecting sudden and gradual drifts using statistical testing of 
feature vectors. Although this approach is pioneering, it has limitations in automated 
drift detection. Users must specify the type of drift in advance and manually select fea -
tures, relying on prior knowledge of drift characteristics. Otherwise, they may face the 
computational burden of testing all feature combinations. Additionally, users must specify a window size for drift detection, potentially missing some drift occurrences.
To address the window size limitation, Martushev et  al. (2015) enhanced this 
method by introducing adaptive windowing, which automatically adjusts the window 
size when searching for drifts. However, this approach still requires users to define 
minimum and maximum window size parameters as boundaries for automated win -
dow adaptation. Moreover, users must define the type of drift beforehand. Therefore, 
Table 1 Classification of different concept drift detection techniques
Legend: “✷” - automated, “(✷)” - semi-automated
Technique Change point 
detection
Drift type detection
Sudden Gradual Incremental Recurring
Various works ✷
Bose et al. (2013) ✷ (✷) (✷)
Martjushev et al. (2015) ✷ (✷) (✷)
Maaradji et al. (2017) ✷ ✷ ✷
Yeshchenko et al. (2021) ✷ ✷ (✷) (✷) (✷)
Our approach (CV4CDD-4D) ✷ ✷ ✷ ✷ ✷
1 Recent techniques (Neto et al. 2023; Meira Neto et al. 2024) demonstrate potential for detecting all four drift types. 
However, their current implementations are limited to change point detection, with plans to extend their capabilities to 
other drift types in future versions.

## Page 6 / 26

Page 6 of 26Kraus and van der Aa Process Science (2025) 2:5 
both techniques remain limited to detecting only sudden and gradual drifts, neglect -
ing recurring and incremental drifts.
The work by Maaradji et al. (2017) introduced an alternative technique (ProDrift) 
for detecting sudden and gradual drifts, addressing the main limitation of previous 
solutions (Bose et al. 2013; Martjushev et al. 2015) and representing the current state 
of the art in simple drift detection. Their approach involves a two-step process: first, 
it detects change points to identify sudden drifts and then it employs post-process -
ing on the output of the first step to detect gradual drifts. Specifically, they analyze 
the behavior within the intervals between two change points by statistically assessing 
whether it exhibits a mixture of behavior distributions before and after these points. 
However, this distribution-based method has a significant drawback: in  situations 
where noise is present in an event log, their approach experiences a notable decrease 
in detection accuracy, as demonstrated in our evaluation (see the  “Experiment 1: 
change point detection ” section).
The Visual Drift Detection (VDD) technique, introduced by Yeshchenko et al. (2021), 
is a stand-alone solution for detecting four types of concept drifts in event logs. This 
technique leverages concepts such as temporal logic, Declare constraints (Ciccio and 
Mecella,  2015), and time series analysis to group similar declarative behavioral con -
straints and automatically identify change points. To identify different types of drifts, 
it provides visual aids, including Drift Maps, Drift Charts, and directly-follows graphs. 
However, the identification of gradual, recurring, and incremental drifts remains a 
manual process that relies on user interpretation. As a result, recognizing drift and their 
types within the same event log can be challenging and subjective. Our approach over -
comes this problem by detecting all four types of drifts in an automated manner.
Overall, it is evident that the comprehensive offline detection and characterization 
of drifts and their types in event logs have not been adequately addressed.
Other concept drift detection techniques. A variety of existing techniques for detect -
ing concept drift address other aspects of the problem. For instance, some approaches 
simultaneously consider multiple process perspectives, such as time, resource involve -
ment, and data, when identifying concept drifts (Klijn et  al. 2024; Qahtan et  al. 2015; 
Richter and Seidl 2017; Cao et al. 2024). Other techniques not only aim to detect con -
cept drifts but also provide explanations for these changes (Adams et  al. 2023; Guan 
et al. 2023). Various works focus on detecting concept drifts in event streams (online settings), which allows for real-time detection. These techniques can detect change points 
(Hassani 2019; Ostovar et  al. 2016) and offer detailed drift localization (Ostovar et  al. 
2017, 2020) while dealing with challenges such as computational overhead, real-time 
processing requirements, and the need for continuous monitoring. Some online drift 
detection techniques specifically target drift type detection (Stertz and Rinderle-Ma 
2018; Richter and Seidl 2017; Barbon Junior et al. 2018; Tavares et al. 2019; Huete et al. 
2023; Hanga et  al. 2022), while also considering different process perspectives (Stertz 
and Rinderle-Ma, 2019). Finally, certain techniques operate on trace streams rather than 
event streams. These techniques can identify change points (Lakshmanan et  al. 2011; 
Carmona and Gavaldà 2012; Zellner et al. 2021; Weber et al. 2011; de Sousa et al. 2021; 
Sato et al. 2021) and detect drift types (Liu et al. 2018).

## Page 7 / 26

Page 7 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
In the following section, we present our approach, which overcomes existing limita -
tions and advances the state of the art by offering a reliable and automated method for 
identifying all four types of drifts in event logs. Our method leverages an innovative 
machine learning paradigm to learn how drifts manifest in event logs, moving beyond 
traditional hand-crafted, unsupervised techniques.
Approach
This section introduces CV4CDD-4D, our computer vision approach for concept drift 
detection. As visualized in Fig.  2, our approach consists of two main steps. First, we 
transform an event log into an image that captures the behavioral (dis)similarity of a 
process over time. Then, the image is passed to our fine-tuned computer vision model, 
which identifies if drifts are present in an event log and, if so, determines their type (sudden, gradual, incremental, recurring) and corresponding change points.
Before describing the two steps of our approach in detail, we define the approach’s 
input and desired outcome.
Input. Our approach takes as input an event log L, which we define as a collection of 
events e recorded by an information system during process execution. Each event e ∈ L 
needs to at least have a case ID, an activity, and a timestamp. We use σ to represent a 
trace, a sequence of events from L with the same case ID, ordered by their timestamps. 
Finally, we denote /Sigma1L as the ordered collection of traces, arranged according to the 
timestamp of their first event.
Output. Given an event log L as input, the desired output is a collection of drifts 
D := {d 1, ··· , dn} , where each drift di is represented by a tuple, di := (type, pstart , pend ) , 
with di.type specifying the drift type, and di.pstart and di.pend denoting the drift’s start 
and end change points, respectively. In case of a sudden drift, the start and end points 
are the same. For a complex drift, which consists of a sequence of sudden and gradual 
drifts, the start and end change points are defined by the start change point of the first 
drift and the end change point of the last drift in the sequence, respectively.
Transformation of event log into image
The first approach step takes as input an event log and transforms it into an image. The 
image visualizes the behavioral (dis)similarity of a process over time recorded, which 
can be used to recognize concept drifts. The transformation includes four sub-steps, as 
depicted in Fig. 3.
Split traces into windows. The approach first splits the chronologically ordered traces 
in /Sigma1L into an ordered collection of N windows, W := �w 1, ... , wN � . These windows are 
Fig. 2 Our CV4CDD-4D approach: overview of the main steps

## Page 8 / 26

Page 8 of 26Kraus and van der Aa Process Science (2025) 2:5 
non-overlapping and collectively cover all recorded traces in /Sigma1L , with each window wi 
containing approximately |� L|/N traces.
During the fine-tuning of the computer vision model, we use N = 200 as a default 
value for the number of windows. For the training collection (see Table 2), we tested various options and selected 200 because it, on average, provides an effective image-based 
representation for event logs in the context of concept drift detection. This choice was 
motivated by the need for a window size that strikes a balance: it should not be too large 
(which could conceal gradual drifts within a single window, making them undetectable) 
or too small (which might capture too few process traces to adequately represent pro -
cess behavior). The selection is primarily influenced by the configurations for the length 
of generated drift and drift-free periods (and their more complex patterns) used to gen -
erate event logs for our training collection. Therefore, using a different configuration or 
data generation tool would likely require re-evaluating the choice of this parameter.
We recommend using 200 windows also as the default setting for detecting concept 
drifts using CV4CDD-4D on a new event log. However, for small event logs (e.g., with 
fewer than 2,000 traces), decreasing the number of windows avoids having too few traces 
per window, as demonstrated in our sensitivity analysis in Experiment 1 (see the “Experiment 1: change point detection” section). Conversely, larger event logs (especially those 
spanning a long time range) may benefit from having more windows, to prevent drifts 
occurring within the span of a single window.
Calculate behavioral representation. After establishing W, our approach computes 
a behavioral representation, B(wi) for each window wi , which characterizes the process 
behavior of wi ’s traces. Each B(wi) consists of a collection of two-dimensional tuples, 
each storing a behavioral pattern and its frequency, as visualized in Fig. 4.
A common behavioral representation used in process mining is to capture the directlyfollows frequencies observed during a time window (Elkhawaga et  al. 2020), which we 
use as the default for our approach. It counts how often an activity was observed to 
directly succeed another one in all the traces of a given window. However, it is important 
Fig. 3 First approach step: transforming an event log into an image
Fig. 4 Illustration of the similarity calculation between two windows

## Page 9 / 26

Page 9 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
to note that the choice for a behavioral representation is flexible, provided that it yields 
a numeric frequency distribution over a predefined set of relations or patterns across 
the window. Therefore, CV4CDD-4D can also cover other types of relations (e.g., eventually follows), sets of relation types, such as those of a behavioral profile (Smirnov et al. 
2012), or declarative constraints (van Der Aalst et al. 2009). However, this would require 
retraining the computer vision model to learn drift patterns in the new behavior repre -
sentation (see the “Drift detection” section).
Measure similarity. Next, our approach compares the behavioral representations 
obtained for the different windows, quantifying their similarity. This comparison is done 
for each pair wi and wj from W, resulting in a symmetric similarity matrix denoted as S. 
Each entry S[i, j] in this matrix shows the similarity between the behavior represented by 
B(wi) and B(wj ).
The similarity matrix S can be established using various similarity measures. In our 
approach, we use cosine similarity as the default measure because it effectively detects 
newly introduced or removed behavior relations and is well-suited for assessing changes 
in the frequencies. In theory, other commonly used similarity measures, such as Earth 
Mover’s Distance (Brockhoff et  al. 2020), could also be applied. However, this would 
again require retraining the computer vision model. Figure  4 illustrates the calculation 
of the similarity measure between two windows, using a behavioral representation based 
on directly-follows frequencies and the cosine similarity measure.
Transform into image. Finally, to enable image-based concept drift detection, the 
similarity matrix S is transformed into an image. For this transformation, our approach 
normalizes the matrix values to a range between 0 and 1, where the maximum similarity 
corresponds to 1 and the minimum similarity is 0. Each normalized value is then scaled 
by 255 and converted to integers, resulting in a range between 0 and 255. Finally, using 
the Python Imaging Library 2 and a color map, images are generated from the normal -
ized values.
Figure 5 depicts a few examples of the outcome of this step, covering different drift 
scenarios. Note that the annotation that is shown is covered in the  “Drift detection ” 
section.
Fig. 5 Output of the first approach step (incl. annotations)
2 Available online: https:// python- pillow. org

## Page 10 / 26

Page 10 of 26Kraus and van der Aa Process Science (2025) 2:5 
Drift detection
The second step of CV4CDD-4D takes as input the image obtained from the previous 
step and applies a fine-tuned object detection model to detect concept drifts. In this section, we present an overview of the object detection task and employed object detec -
tion model, elaborate on the training data and its annotation, and clarify the training 
configurations.
Object detection using RetinaNet. Object detection is a fundamental task in com -
puter vision, where the goal is to identify and locate objects within images. Deep learn -
ing methods have significantly advanced this field by directly learning features from 
data, leading to breakthroughs in object detection (Liu et al. 2020). RetinaNet (Lin et al. 
2017) is a recent addition to deep learning-based object detection models. Known for its 
effectiveness and reliability, RetinaNet has become widely adopted in both research and 
practical applications, setting new standards in object detection performance.
RetinaNet Architecture. The RetinaNet model consists of three main components 
(Lin et al. 2017):
• Backbone Network: The backbone network identifies patterns necessary for object 
detection. RetinaNet uses ResNet, a deep convolutional neural network, which 
improves learning efficiency by allowing inputs to bypass certain layers.
• Feature Pyramid Network (FPN): The FPN processes multi-scale feature maps from 
the backbone to create a pyramid of features at different resolutions. It enhances the 
model’s ability to detect objects of varying sizes by combining high-resolution feature 
maps from earlier layers with lower-resolution feature maps from deeper layers.
• Classification and Bounding Box Subnetworks: These subnetworks enable accu -
rate object detection. The classification subnet predicts the probability of an object 
belonging to a particular class at each location on the feature map, operating at mul -
tiple levels of the FPN. The box regression subnet predicts the coordinates of bounding boxes for objects and refines their positions.
Additionally, RetinaNet uses a specialized loss function, Focal Loss, to address class 
imbalance by down-weighting easily classified background examples and focusing on 
difficult-to-detect objects. This makes RetinaNet effective in detecting concept drift and 
handling noise across various sections of an image.
Training data and annotation. We use a training set of event logs with known con -
cept drifts. Each event log in the training set is transformed into an image using the first 
step of our CV4CDD-4D approach (see the “Transformation of event log into image” section). Then, each image is annotated based on the drift information stored in the gold 
standard, capturing where different drifts occur and what their types are. For the anno -
tation, we define bounding boxes using the widely-employed COCO (Common Objects 
in Context) format (Lin et al. 2014).
In our case, we use these bounding boxes to capture where drifts of certain types 
occurred in the image, as shown for various scenarios in Fig.  5. To annotate sudden 
drifts, characterized by a single change point p that belongs to a window wi , we estab -
lish a bounding box around wi that spans in total 2 windows in both directions from wi , 
resulting in a total length of 5 windows. For gradual drifts, we create annotations using

## Page 11 / 26

Page 11 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
windows that correspond to the start and end change points. Each change point pstart 
and pend is associated with a trace index, which belongs to a particular window in W. 
The corresponding windows wi and wj , allow us to link pstart and pend to their window 
indices i and j. We use these indices to define a bounding box for gradual drift within an 
image.
Training configurations. To operationalize CV4CDD-4D, we specifically use the RetinaNet model from the TensorFlow Model Garden3, based on the SpineNet backbone (ID 
143). The model is pre-trained on the COCO dataset 4, a widely-used dataset for object 
detection. To adapt it to our specific task, we fine-tune the model on a training set and 
halt fine-tuning using a validation set to prevent overfitting.
Image input, batch size, anchor boxes. We use a fixed input size of 256 ×256 for finetuning the model5 with a batch size of 128 images, taking 312 iterations per epoch. The 
model undergoes 500 epochs of training, with augmented images to increase diversity 
and robustness by scaling up to two times or down to one-tenth of their original size. We 
employ anchor boxes with a 1:1 aspect ratio, concentrating exclusively on square-shaped 
bounding boxes, since all annotations correspond to squares of different sizes along the 
diagonal of the image.
Optimization and learning rate. We use stochastic gradient descent with a momentum 
of 0.9 and clip norm of 10, known for its simplicity and efficiency in training deep learn -
ing models, especially with large datasets. Momentum aids convergence by leveraging 
past gradients, while gradient clipping prevents exploding gradients, ensuring stability, 
particularly in complex neural networks. Additionally, we employ a cosine learning rate, 
widely adopted for its simplicity and ability to enhance convergence and generalization. 
This schedule adjusts the learning rate throughout training, following a cosine-shaped 
function.
Confidence level. For each detected drift, the model assigns a confidence level between 
0 and 1, reflecting its certainty in identifying and classifying the drift in the image. A 
score closer to 0 indicates that no drift is present, while a score closer to 1 means the 
model is confident that a drift of a certain type is present and correctly classified. We set 
a threshold of 50% for the detection confidence level, meaning that if the model detects a 
drift type with confidence above 50%, we consider the drift as detected.
Using the fine-tuned RetinaNet model, our CD4CDD-4D approach can be directly 
applied to detect concept drifts in unseen event logs.
Evaluation
This section presents three experiments conducted to comprehensively evaluate the performance of our CV4CDD-4D approach for concept drift detection from multiple per -
spectives. In Experiment 1, we assess the accuracy of our approach in detecting change 
points in event logs and compare our results to various baseline techniques that address 
this critical task for concept drift detection. Next, in Experiment 2, we evaluate the 
3 TensorFlow Model Garden, Available online: https:// github. com/ tenso rflow/ models
4 COCO dataset, Available online: https:// cocod ataset. org/
5 If an input image provided for inference has a different pixel size, i.e., because it was established using a different num -
ber of windows N, RetinaNet automatically rescales the image to the default size.

## Page 12 / 26

Page 12 of 26Kraus and van der Aa Process Science (2025) 2:5 
accuracy of our approach in detecting drifts and their types and highlight its advantages 
over a comparable state-of-the-art technique. Finally, in Experiment 3, we apply our 
approach to real-life event logs and compare the insights obtained with findings from 
the state-of-the-art technique.
To ensure reproducibility, the data collection, implementation, configurations, and 
raw results are accessible in our public repository6.
Experiment 1: change point detection
In this section, we assess the ability of our approach to detect change points in event logs 
in comparison to existing baselines using synthetic data. We consider this problem in 
isolation from the detection of drifts, given its fundamental role in concept drift detec -
tion, as also evidenced by the various techniques that have been proposed to address it. 
In the following, we discuss the evaluation setup, obtained results, and findings from a 
sensitivity analysis.
Evaluation setup
Below we elaborate on the details of the data collection, baselines, evaluation measures, 
as well as configurations used to evaluate our approach.
Data collection. Our data collection comprises two datasets, summarized in Table 2.
CDLG dataset. To train, validate, and test our drift detection approach, we require a 
large collection of event logs that contain known (i.e., gold-standard) concept drifts of 
sudden, gradual, incremental, and recurring types. Since such a collection is not publicly 
available, we, therefore, generated synthetic datasets using CDLG (Concept Drift Log 
Generator) (Grimm et al. 2022), a tool for the automated generation of event logs with 
concept drifts, which comes with a wide range of parameters.
We used CDLG to generate 50,000 event logs, allocating 80% for training, 5% for vali -
dation, and 15% testing. The generated event logs have the following characteristics:
Table 2 Characteristics of the two synthetic datasets
Characteristics CDLG dataset CDRIFT
(Number of) Training Validation Test Dataset
Event logs 40000 2500 7500 115
→ without drifts 9834 586 1834 0
→ with noise 19908 1250 3768 60
Drifts 112660 7069 21229 156
→ Sudden drifts 41295 2587 7827 156
→ Gradual drifts 41395 2615 7776 0
→ Incremental drifts 14967 986 2823 0
→ Recurring drifts 15003 881 2803 0
Change points 120151 7501 22567 156
→ Sudden 59971 3726 11333 156
→ Gradual 60180 3775 11234 0
6 Project repository: https:// gitlab. uni- mannh eim. de/ proce ssana lytics/ cv4cdd.

## Page 13 / 26

Page 13 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
• The logs are generated from process trees containing between 6 and 20 activities, 
as well as sequential, choice, parallel, and loop operators.
• Each event log has between 1,000 and 21,000 traces (with an average of around 
7,200 traces per log), and the average trace length varies from 1 to 65 events.
• The event logs have 0 to 3 drifts each (with equal probability). Incremental and 
recurring drifts consist of 3 simple drifts (of sudden or gradual type), yielding a 
maximum of 18 change points per log.
• Each drift introduces changes up to 30% of the process tree elements (activities 
and operators) through deletion, insertion, or swapping.
• A quarter of the logs contain randomly inserted noise in 30% of the traces and 
another quarter in 60% of the traces. The other half are noise-free.
CDRIFT dataset. To assess the generalizability of our approach and verify that its 
performance is not restricted to the characteristics of the CDLG dataset, we also con -
sider a dataset used in a recent experimental study (Adams et  al.  2023), which we 
refer to as the CDRIFT dataset. This set consists of 115 synthetic event logs, previ -
ously employed in evaluating various concept drift detection techniques, stemming 
from three sources (Bose et al. 2013; Ostovar et al. 2016; Ceravolo et al. 2022). The 
logs have about 1700 traces on average and contain between 1 and 4 change points. 
Notably, the CDRIFT dataset only contains sudden drifts, though.
Baselines. We compare our approach to seven techniques for the detection of 
change points that were used in a recent benchmark study by Adams et al. ( 2023): 
1. Bose/J by Bose et  al. ( 2013) uses non-overlapping and continuous fixed-size win -
dowing with activity pair-based feature extraction and statistical testing.
2. Adwin/J by Martjushev et al. ( 2015) improves the Bose/J technique by introducing 
adaptive windowing using the ADWIN approach.
3. ProGraphs by Seeliger et  al. ( 2017) implements non-overlapping and continuous 
adaptive-size windowing, uses graph-based process features alongside Heuristics 
Miner, and employs statistical testing.
4. ProDrift by Maaradji et al. ( 2017) employs non-overlapping continuous fixed and 
adaptive-size windowing, statistical testing, and an oscillation filter.
5. Rinv by Zheng et  al. ( 2017) uses behavioral profiles, a process similarity measure, 
and DBSCAN clustering.
6. EMD by Brockhoff et al. (2020) employs a sliding window approach with local multiactivity feature extraction and the Earth Mover’s Distance.
7. Lcdd by Lin et al. (2020) uses both static and adaptive sliding windows, incorporates 
directly-follows relations, and ensures local completeness.
Evaluation measures. We report on results obtained using established evaluation 
measures for detecting change points (Adams et al. 2023). Specifically, for each event 
log, we compare the sequences of detected Pd =
⣨
pd
1 , ... , pd
n
⟩
 and gold-standard 
Pg = pg
1, ... , pg
m change points ( n, m ≥ 0 ), where each change point is represented 
by the ordinal number of the first trace that started after the change.

## Page 14 / 26

Page 14 of 26Kraus and van der Aa Process Science (2025) 2:5 
To identify which gold-standard change points have been successfully detected, we 
use the linear program proposed by Adams et al. (2023) to establish a pairwise mapping 
between the points in Pd and Pg . This program finds an optimal mapping M, assigning 
as many points to each other as possible, while minimizing the distance between corre -
sponding change points. Note that no point in Pd is assigned to multiple points in Pg or 
vice versa. Furthermore, M will only include pairs pd
i ∼ pg
j that are within an acceptable 
distance from each other, which we refer to as the allowed latency. We define latency as 
a percentage of the total traces in /Sigma1L , i.e., it must hold that |pd
i − pg
j |≤| /Sigma1L|∗latency . 
We report on results obtained using latency levels of 1%, 2.5%, and 5%.
Due to the consideration of latency, each correspondence in M is regarded as a true 
positive. From this, we derive precision (Prc.) as |M|/|Pd | , i.e., the fraction of detected 
change points that are correct according to the gold standard, recall (Rec.) as |M|/|Pg | , 
i.e., the fraction of correctly detected gold-standard change points, and the F1-score as 
the harmonic mean of precision and recall.
Configurations. For different datasets, we use different configurations for the base -
lines and our approach.
Baselines. When reporting on the performance of the baseline techniques, we use the 
parameter settings that we found to achieve the highest F1-score. To find these settings 
for the CDLG dataset, we applied the experimental framework by Adams et al. (2023), 
which assesses different parameter configurations, using the validation subset of the 
CDLG dataset. For the CDRIFT dataset, we ran experiments using all configurations 
that are tested in the Adams et al. (2023) framework and report on the results obtained 
using the best parameter settings. The exact parameter settings used for the different 
techniques per dataset are detailed in our repository.
Our approach. We fine-tuned the RetinaNet model used by our approach with the 
training and validation subsets of the CLDG dataset and parameters described in the 
second step of our approach. Given such a fine-tuned model, the only parameter to set 
for inference is the number of windows N to be used. For the test subset of the CDLG 
dataset, we use the same number of windows as used during the fine-tuning ( N = 200 ). 
For the CDRIFT dataset, similar to the baseline’s parameters, we report on the results 
obtained for the best parameters ( N = 70 ) derived from a sensitivity analysis (see 
the “Sensitivity analysis” section).
Results
In the following, we present the results obtained for the two datasets, also focusing on 
different latency and noise levels7 (Table 3).
Accuracy. Table 3 describes the results for each of the two datasets.
CDLG dataset (test). For the test subset of the CDLG dataset, our CV4CDD-4D 
approach consistently outperforms the baselines, demonstrating F1-scores ranging from 
0.81 at 1% latency to 0.83 at 2.5% and 5% latencies. It already reaches its peak perfor -
mance at just 2.5% latency, surpassing the best baseline, EMD, by 0.27. In terms of recall, 
7 Given the non-determinism involved in training deep learning models, we repeated the training and inference pro -
cedure of our approach five times. These repetitions resulted in mean standard deviations of less than 0.1 percentage 
points across all measures (for the test subset of the CDLG dataset). We report on the results of the first run in the 
remainder.

## Page 15 / 26

Page 15 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
our approach outperforms the baseline scores by 0.30 and 0.15 at 1% and 2.5% laten -
cies, respectively. At 5% latencies, our approach achieves also the highest recall of 0.77, 
however, the Lcdd , EMD, and Bose/J techniques achieve comparable recall scores, each 
exceeding 0.70. Despite this, they exhibit lower precision, resulting in significantly lower 
F1-scores. Finally, in terms of precision, our CV4CDD-4D approach surpasses the bestperforming baseline, ProDrift , by margins of 0.22, 0.16, and 0.14 for 1%, 2.5%, and 5% 
latency, respectively.
CDRIFT dataset. We obtain overall similar results for the CDRIFT dataset. Our 
CV4CDD-4D approach outperforms all baselines across latency levels, achieving 
F1-scores of 0.56, 0.92, and 0.92 at 1%, 2.5%, and 5% latencies, respectively. These higher 
values can be attributed to the fact that the CDRIFT dataset contains only sudden drifts, 
which are relatively easier to detect for our approach. Only at 1% latency does ProDrift 
achieve a higher precision of 0.91 compared to 0.61 for our approach. The reason for the 
lower precision of our approach is rather technical. It is mainly attributed to the annotation of sudden drifts using bounding boxes of 5 windows spanning around the position 
of an actual sudden drift in an image. In scenarios with 70 windows and a latency of 
just 1%, inaccuracies arise during the transformation from the coordinates of the bounding box to the corresponding window index and subsequently to the first trace within 
the window, leading to low precision and recall. This is supported by the correctly posi -
tioned bounding boxes in the respective images, along with the observation that accu -
racy sharply increases to its peak values at the next latency of 2.5%.
Noise impact. To evaluate the robustness to noise of our approach, we report the 
results for the event logs with different noise levels in the test subset of the CDLG data -
set. We report results using 5% latency because the accuracy of several baselines drops 
significantly for 1% and 2.5% latencies, making it difficult to determine whether the 
decrease in performance is caused by noise or latency.
Table 3 Overall results for detecting change points
Support: CDLG dataset (test): 22567 change points, CDRIFT dataset: 156 change points
Dataset Technique Latency 1% Latency 2.5% Latency 5%
Prc. Rec. F1 Prc. Rec. F1 Prc. Rec. F1
CDLG (test) Bose /J 0.32 0.39 0.25 0.49 0.60 0.54 0.57 0.70 0.63
Adwin /J 0.43 0.28 0.34 0.58 0.38 0.46 0.63 0.42 0.50
Pro GrAPhs 0.24 0.26 0.25 0.48 0.52 0.50 0.58 0.64 0.61
Pro d rift 0.55 0.22 0.32 0.74 0.30 0.43 0.76 0.31 0.44
r inv 0.36 0.44 0.40 0.46 0.56 0.51 0.53 0.64 0.58
eMd 0.36 0.44 0.39 0.51 0.62 0.56 0.58 0.71 0.64
Lcdd 0.27 0.41 0.32 0.39 0.61 0.48 0.46 0.72 0.56
CV4CDD-4D 0.87 0.75 0.81 0.90 0.77 0.83 0.90 0.77 0.83
CDRIFT Bose /J 0.08 0.07 0.07 0.60 0.52 0.56 0.75 0.66 0.70
Adwin /J 0.15 0.11 0.13 0.40 0.29 0.34 0.71 0.51 0.59
Pro GrAPhs 0.21 0.18 0.19 0.48 0.41 0.45 0.78 0.67 0.72
Pro d rift 0.91 0.32 0.48 1.00 0.35 0.52 1.00 0.35 0.52
r inv 0.01 0.00 0.00 0.23 0.18 0.20 0.47 0.36 0.41
eMd 0.05 0.03 0.04 0.88 0.59 0.71 0.97 0.66 0.79
Lcdd 0.00 0.01 0.01 0.02 0.04 0.02 0.24 0.64 0.35
CV4CDD-4D 0.61 0.52 0.56 1.00 0.86 0.92 1.00 0.86 0.92

## Page 16 / 26

Page 16 of 26Kraus and van der Aa Process Science (2025) 2:5 
As summarized in Table  4, our CV4CDD-4D approach maintains consistent perfor -
mance regardless of noise, achieving the highest F1-scores from 0.83 for logs without 
noise to 0.82 and 0.81 for logs with 30% and 60% noisy traces, respectively. In noisefree conditions, three baselines (ProDrift , Lccd , and Rinv ) come close to our results. 
The Rinv technique shows an outstanding recall of 0.83, while ProDrift maintains its 
lead across the baselines in precision. However, all three of these baselines experience 
a notable decline in accuracy when noise is introduced, particularly ProDrift . Con -
versely, baselines with relatively lower accuracy on the noise-free logs (EMD , Adwin/J, 
and ProGraphs ) demonstrate less vulnerability to noise. This reveals that the baselines 
are subject to a trade-off between performance in noise-free conditions and robustness 
to noise, which does not apply to our approach.
Sensitivity analysis
Finally, we discuss how the number of windows, N, specified by the user affects the per -
formance of our approach. To investigate this, we conduct a sensitivity analysis that 
examines a range of windows and the corresponding evaluation measures for the two 
datasets. Considering the average size of the event logs, we analyze windows between 
100 and 300 for the test subset of the CDLG dataset and between 60 and 200 for the 
CDRIFT dataset.
CDLG dataset (test). Figure  6 shows the effects of different number of windows on 
the evaluation measures across the latency levels. All three figures show the same major 
trend. Specifically, we observe that the F1-score remains within a corridor of ±5 percentage points for each latency level, with a slight decline in performance at both extremes 
Table 4 Noise impact on detecting change points (5% latency)
Support: 22567 change points
Technique W/o noise With 30% noise With 60% noise
Prc. Rec. F1 Prc. Rec. F1 Prc. Rec. F1
Bose /J 0.59 0.70 0.64 0.57 0.70 0.63 0.55 0.71 0.62
Adwin /J 0.64 0.43 0.51 0.62 0.41 0.50 0.62 0.40 0.49
Pro GrAPhs 0.55 0.64 0.59 0.60 0.66 0.63 0.64 0.61 0.63
Pro d rift 0.76 0.60 0.67 0.67 0.002 0.004 0.33 0.004 0.001
r inv 0.63 0.83 0.72 0.39 0.41 0.40 0.40 0.48 0.44
eMd 0.58 0.72 0.64 0.59 0.70 0.64 0.57 0.70 0.62
Lcdd 0.63 0.76 0.69 0.35 0.77 0.48 0.36 0.60 0.45
CV4CDD-4D 0.89 0.78 0.83 0.89 0.75 0.82 0.89 0.75 0.81
Fig. 6 Sensitivity analysis (the test subset of the CDLG dataset)

## Page 17 / 26

Page 17 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
of the examined range of windows. This indicates that the approach is generally robust 
to the choice of the number of windows when detecting drifts in the test subset of the 
CDLG dataset.
CDRIFT dataset. Figure  7 illustrates the results of our sensitivity analysis, where we 
can see two major findings. First, the results suggest that for event logs with relatively 
few traces (as in the CDRIFT dataset), it is reasonable to reduce the number of win -
dows from the default value of 200 to 100 or fewer, allowing each window to capture 
more traces and better represent process behavior. Second, a noticeable decline in recall 
occurs between 100 and 160 windows, with the lowest performance observed at 130 
windows. This outcome can be attributed to the structure of the synthetic event logs in 
the CDRIFT dataset, which typically feature either one change point in the middle of the 
log (in the majority of logs) or multiple evenly spaced change points. At 130 windows, 
change points fall near the center of a window, causing the distance from the start of the 
window (used for the detection of change points) to exceed the allowable latency, resulting in reduced performance.
Overall, we can observe in this experiment that our approach achieves a notable per -
formance improvement with respect to latency and shows consistent robustness to noise 
when it comes to the detection of change points in event logs.
Experiment 2: concept drift detection
This section discusses the experiment conducted to evaluate the performance of our 
approach to detect drift and their types, also in comparison with the state-of-theart technique. In the following, we discuss the evaluation setup, obtained results, and 
insights from a sensitivity analysis.
Evaluation setup
First, we provide information regarding the baseline, data collection, configurations, and 
evaluation measures used in this experiment.
Baseline. We compare our approach against the Visual Drift Detection ( VDD) technique proposed by Yeshchenko at el. (2021). We selected this technique because it stands 
out as the only existing technique that can be used to detect four types of drifts from an 
event log and is the only (partially) comparable solution to our approach, due to a lag in 
automation. In our experiments, we use the online version of the VDD technique8.
Fig. 7 Sensitivity analysis (CDRIFT dataset)
8 Available online: https:// yesan ton. github. io/ Proce ss- Drift- Visua lizat ion- With- Decla re/ client/ build/

## Page 18 / 26

Page 18 of 26Kraus and van der Aa Process Science (2025) 2:5 
Data collection. We use the test subset of the CDLG dataset to evaluate the accuracy of 
our approach in detecting concept drift, as no other collections of event logs encompass the 
necessary drift scenarios that include a mix of different numbers, types of drifts, noise levels, and severities of process changes. However, since the VDD technique is not fully automated and depends on user interpretation of visualizations, we showcase the advantages of 
our approach using a specific event log from our test subset of the CDLG dataset. Specifically, we use log number 5436, which describes a complex drift scenario with noisy behavior 
and serves as a proper example to demonstrate the effectiveness of our approach in detecting complex drifts, particularly in comparison to the baseline. This log contains 60% noise, 
72,433 events, 10,103 traces, 3,787 trace variants, and 9 distinct activities. The left side of 
Table 7 indicates that the log includes two complex drifts: incremental and recurring. Both 
drifts consist of a sequence of three simple drifts: sudden, gradual, and sudden, leading to a 
total of 8 change points.
Configurations. For the baseline, we use the suggested default parameters of the online 
version of the tool for the selected event log: window size 330, slide size 165, cut threshold 
300. For our CV4CDD-4D approach, we use the default value of 200 windows.
Evaluation measures. We report on precision, recall, and F1-score by comparing a collection of detected drifts Dd to the gold-standard drifts Dg . For each drift type, t, and the 
corresponding detected Dd (t) ⊆ Dd and gold-standard drifts Dg (t) ⊆ Dg , a true positive 
(tp) occurs if there is a detected drift dd
k ∈ Dd (t) of which both the start and end change 
points correspond to those of a gold-standard drift dg
l ∈ Dg (t) (given a certain latency 
level). However, if only the start or end point of dg
l is detected correctly, we still count it 
as 0.5 of a true positive (as well as 0.5 of a false positive). If neither of the detected change 
points for a drift corresponds to the gold standard, it is considered a false positive (fp ). 
Finally, the number of false negative and true positives ( fn + tp ) is given by the number of 
actual drifts of a given type.
Given these scores, we compute precision (Prc.) as tp/(tp + fp) , recall (Rec.) as 
tp/(fn + tp) , and the F1-score per drift type (and logs without drifts), as well for the overall 
detection (using weights to account for their different support values).
Results
In the following, we present the overall results of our approach with respect to different 
latency and noise levels. Then, we show the advantage of using our approach in comparison to the baseline.
Table 5 Concept drift detection results by latency levels
Drift Support Latency 1% Latency 2.5% Latency 5%
Prc. Rec. F1 Prc. Rec. F1 Prc. Rec. F1
No drifts 1834 0.91 1.00 0.95 0.91 1.00 0.95 0.91 1.00 0.95
Sudden 11333 0.99 0.76 0.86 0.99 0.76 0.86 0.99 0.76 0.86
Gradual 11234 0.99 0.60 0.75 1.00 0.62 0.77 1.00 0.62 0.77
Incremental 2823 0.97 0.92 0.95 0.98 0.97 0.98 0.99 0.98 0.98
Recurring 2803 0.99 0.97 0.98 0.99 0.99 0.99 0.99 0.99 0.99
Overall 0.98 0.75 0.84 0.99 0.76 0.86 0.99 0.76 0.86

## Page 19 / 26

Page 19 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
Accuracy. Table  5 presents the results obtained for different latency levels. Our 
CV4CDD-4D approach shows F1-scores ranging from 0.84 at 1% latency to 0.86 at 5% 
latency. For event logs without any drifts, the approach achieves a perfect recall of 1.00 
and a precision of 0.91 across all latency levels. This indicates that it correctly identi -
fies all event logs without drifts. However, in 1 out of 10 cases, the approach incorrectly 
detects a drift in an event log where no drift exists. For sudden and gradual drifts, pre -
cision remains high (above 0.99), but recall drops to 0.76 for sudden drifts and ranges 
between 0.60 and 0.62 for gradual drifts. This suggests that some actual sudden and 
gradual drifts, particularly gradual ones, are not detected as such. This can be attributed 
to the fact that our test set includes event logs with varying noise levels and process 
changes of different severities, making accurate detection a challenging task. Lastly, for 
incremental and recurring drifts, the approach achieves results above 0.92 for all meas -
ures and latency levels, indicating that it accurately detects the start and end points of 
these more complex drifts. Compared to simple drifts, the recall for complex drifts is 
notably higher. This is due to the more distinct patterns these drifts produce in an image, 
which makes them easier for the approach to detect. In contrast, simple drifts are more 
likely to be missed, particularly when the change severity is low or noise is present.
Noise impact. Table 6 shows the results for different noise levels. Overall, the results 
remain consistent across all noise levels, demonstrating that our approach is robust to 
noise. The only notable variation is a slight decrease in recall for gradual drifts, from 
0.63 for noise-free event logs to 0.59 for logs with 60% of noise. This suggests that the 
Table 6 Concept drift detection results by noise levels
Drift Support W/o noise With 30% noise With 60% noise
Prc. Rec. F1 Prc. Rec. F1 Prc. Rec. F1
No drifts 1834 0.91 1.00 0.95 0.92 1.00 0.96 0.90 1.00 0.95
Sudden 11333 0.99 0.76 0.86 0.99 0.77 0.86 0.99 0.75 0.85
Gradual 11234 1.00 0.63 0.77 0.99 0.60 0.75 0.99 0.59 0.74
Incremental 2823 0.98 0.95 0.97 0.99 0.97 0.98 0.98 0.95 0.96
Recurring 2803 0.99 0.98 0.99 0.99 0.98 0.99 0.99 0.98 0.99
Overall 0.99 0.77 0.86 0.99 0.76 0.85 0.98 0.75 0.84
Fig. 8 Drift detection results for the selected log

## Page 20 / 26

Page 20 of 26Kraus and van der Aa Process Science (2025) 2:5 
relatively low recall observed in Table  5 is primarily due to the complexity of the drift 
scenario rather than the influence of noise.
Comparison with the baseline. Figure  8 presents the results of our approach com -
pared to the baseline for the selected event log.
Our approach. Figure  8a illustrates the results of our CV4CDD-4D approach, high -
lighting the detected drifts and their corresponding confidence levels, which indicate the 
probability of the detected drift belonging to a specific drift type. Table  7 shows the corresponding summary of all detected drifts, including their type and respective start and 
end points. Based on the actual drifts and the deviations between detected and actual 
change points, our approach successfully identifies all drifts in this event log for a 1% 
latency, which allows deviations of at most 101 traces, given the log size of 10,103 traces. 
The average deviation across all change points is approximately 23 traces, which is con -
sistent with the expected deviation9.
VDD technique. Figure 8b presents the primary outcome of the baseline technique: the 
Drift Map. This map displays over 900 detected behavioral rules (on the y-axis), organ -
ized into 61 behavioral clusters, which are indicated by horizontal dashed white lines. 
The Drift Map highlights four sudden drifts indicated by vertical dashed black lines. 
While the first and the last drifts are correctly identified, the two other sudden drifts 
actually mark two change points that denote the start and end moments of a gradual 
drift. This gradual drift can only be identified through visual inspection of the gradual 
change in the confidence level in certain behavior clusters. Using the Drift Map, users 
can also observe that several clusters exhibit recurring behavior in the second part of 
the log, suggesting that detected drifts belong to a recurring drift. However, analyzing 
the first part of the log, which contains incremental drifts, proves to be more challeng -
ing. Although the Drift Map detects several changes across different behavioral clusters 
(indicated by a white dashed line within clusters), it becomes nearly impossible to con -
clude that these changes are part of the same incremental drift.
Table 7 Actual drifts vs. detected drifts using our approach
“(+)” indicates that the detected information correspond to the actual, given 1% latency
Actual drifts Detected drifts Deviation
Type Start End Type Start End Start End
Incremental 1387 4668 (+) Incremental (+) 1351 (+) 4651 − 36 − 17
→ Sudden 1387 1387 (+) → Sudden (+) 1351 (+) 1351 − 36 − 36
→ Gradual 2986 3463 (+) → Gradual (+) 2951 (+) 3401 − 35 − 62
→ Sudden 4668 4668 (+) → Sudden (+) 4651 (+) 4651 − 17 − 17
Recurring 5835 9001 (+) Recurring (+) 5801 (+) 9001 − 34 0
→ Sudden 5835 5835 (+) → Sudden (+) 5801 (+) 5801 − 34 − 34
→ Gradual 6947 7321 (+) → Gradual (+) 6951 (+) 7301 4 − 20
→ Sudden 9001 9001 (+) → Sudden (+) 9001 (+) 9001 0 0
9 For a log with 10,103 traces divided into 200 windows, each window contains approximately 50 traces. In cases of 
accurate drift detection, the expected error between the actual trace index and the first trace in the window is half the 
window size, i.e., 25 traces.

## Page 21 / 26

Page 21 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
Sensitivity analysis
Similar to the first experiment, we conclude this experiment by presenting the insights 
gained from the sensitivity analysis conducted on the test subset of the CDLG dataset. 
We consider again different number of windows between 100 and 300 and illustrate the 
impact of the number of windows on drift detection accuracy across three latency levels.
Figure 9 illustrates how varying the number of windows influences evaluation meas -
ures across different latency levels. Similar to the sensitivity results from Experiment 1 
(for the test subset of the CDLG dataset), we again observe a slight decline in perfor -
mance at both extremes of the examined number of windows. However, regardless of 
latency, the evaluation measures remain within a corridor of ±5 percentage points, indicating that drift detection is also robust to the choice of the number of windows.
In summary, the evaluation results of this experiment suggest that our approach 
detects various types of drifts with high precision and acceptable recall across different 
latencies, while remaining robust to noise and choice of the number of windows. Com -
pared to the state-of-the-art techniques, our approach represents a notable advance -
ment towards the automated and thorough detection of concept drifts.
Experiment 3: evaluation on real‑life event logs
In this experiment, we report results obtained using our approach on real-life event logs 
and compare them with findings obtained in a comparable study. In the following, we 
discuss the evaluation setup and obtained results.
Evaluation setup
We provide a summary of the characteristics of the real-life event logs and detail the 
configurations used in our approach.
Data collection. To allow a direct comparison of insights, we selected the same three 
real-life event logs used in the study by Yeshchenko et  al. (2021), where the authors 
Fig. 9 Sensitivity analysis
Table 8 Characteristics of the real-life event logs
Log name #Traces #Trace variants #Events #Unique 
activities
Hospital Log 1,143 981 150,291 624
Help desk Log 4,580 226 21,348 14
Sepsis Log 1,050 846 15,214 16

## Page 22 / 26

Page 22 of 26Kraus and van der Aa Process Science (2025) 2:5 
employed their VDD technique to detect four types of drifts. The characteristics of the 
selected event logs are summarized in Table  8. These logs exhibit diverse characteristics 
in terms of the number of traces, trace variants (unique sequence of executed activities), 
number of events, and distinct activities.
Configuration. To accommodate the relatively low number of traces in the selected 
real-life event logs, we applied our approach using an adjusted number of windows. 
Specifically, we report results using 75 windows for the Hospital and Sepsis logs (which 
contain around 1,000 traces) and 100 windows for the Helpdesk event log (which has 
approximately 4,500 traces).
Results
Figure  10 presents the results obtained by our approach for the real-life event logs, 
which we compare with the findings reported using the VDD technique (Yeshchenko 
et al. 2021).
Hospital Log. For the Hospital event log, our method identifies a single sudden 
drift around June 12, 2006. This detection partially aligns with the second sudden drift 
reported by the VDD technique (around July 07, 2006) (Yeshchenko et al. 2021). How -
ever, the Drift Map identified an additional sudden drift in November 2005, which our 
approach does not capture. Lastly, neither our approach nor the VDD tool detected any 
complex drifts within the log.
Help Desk Log. In the case of the Help desk event log, our approach identifies an 
incremental drift that begins on September 5, 2012, and concludes on July 10, 2013. 
Within this period, we also detect a gradual drift between June 8, 2011, and August 9, 
2013, as part of the incremental drift process. The start points of these two drifts cor -
respond to the sudden drifts detected by the VDD technique. However, according to our 
findings, these drifts represent change points within the detected incremental drift. For 
certain behavioral clusters, the VDD technique also detects gradual and incremental 
drifts, which are temporally aligned with the two drifts identified by our approach. Additionally, the VDD tool detects several sudden drifts in some clusters, though, as noted 
by the authors, these are considered outliers rather than true drifts. Our image-based 
representation of the event logs depicts these outliers, though our approach does not 
classify them as drifts.
Fig. 10 Detected drifts by our CV4CDD-4D approach

## Page 23 / 26

Page 23 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
Sepsis Log. Our approach does not detect any notable drifts in the Sepsis event log. 
As shown in Fig.  10, the process behavior remains relatively homogeneous throughout 
the entire recorded period. Similarly, the VDD technique does not show any patterns 
of change over time, indicating that the major clusters of behavior do not exhibit sig -
nificant drifts (Yeshchenko et  al. 2021). However, the authors identify some recurring 
patterns in a few minor clusters, which are associated with two specific activities. While 
these patterns may indeed suggest recurring drifts, they could also be attributed to the 
specific cases within these minor clusters, as they represent only a small portion of the 
overall behavior.
Overall, this experiment demonstrates that our CV4CDD-4D approach can be effec -
tively applied to real-life event logs, providing insights that align with and also extend 
the findings of the state-of-the-art technique. However, our approach automatically 
identifies when drifts occur and what their types are without the need of any additional 
interpretation of different visualizations.
Conclusion
In this paper, we proposed CV4CDD-4D, a concept drift detection approach that can 
detect sudden, gradual, incremental, and recurring drifts in an automated manner from 
event logs (offline setting). It is based on a novel idea to detect drifts in an event log 
using an object detection model (RetinaNet) fine-tuned with a large collection of event 
logs that contain known concept drifts. In the conducted experiments, we demonstrated 
that CV4CDD-4D considerably outperforms available baselines for detecting change 
points in event logs across several datasets, including well-established datasets com -
monly used to evaluate concept drift detection techniques. We also demonstrated the 
accuracy in detecting all four types of drifts and the robustness to noise of our approach 
to noise. Additionally, we highlighted its advantages in performing qualitative analysis 
on various real-world event logs compared to the state of the art. Finally, it is worth not -
ing that CV4CDD-4D stands out not only as the first approach using techniques from 
computer vision for concept drift detection in process mining, but as the first approach 
using supervised machine learning in general.
In future work, we aim to address the limitations of our approach and enhance its 
capabilities. We plan to refine the annotation of sudden drifts, as the current bound -
ing box of 5 windows may be too large for small event logs. Additionally, our goal is to 
develop an algorithm to determine the optimal number of windows based on event log 
characteristics, removing the need for user selection. We also intend to train our model 
using diverse data sources, moving beyond our current reliance on a single tool for gen -
erating event logs with known concept drifts. We plan to enhance our model by train -
ing it with diverse data sources and various parameter options for similarity measures 
and behavioral representations, thereby improving its sensitivity to drift patterns. We 
will enhance our capabilities in concept drift detection to encompass multiple process 
perspectives, including time, resources, and data. Additionally, we target including drift 
localization to gain insights into the changes that occur after each drift. Finally, we aim 
to extend further the evaluation of our approach using real-life event logs, particularly 
focusing on datasets with known concept drifts as they become available.

## Page 24 / 26

Page 24 of 26Kraus and van der Aa Process Science (2025) 2:5 
Acknowledgements
We acknowledge the work of Jonathan Kößler for conducting the initial testing of the approach’s idea in his master’s 
thesis.
Authors’contributions
Both authors were involved in designing the approach and the evaluation experiments and worked together to write 
the manuscript. A.K. handled the implementation and conducted the evaluation experiments.
Funding
Open Access funding enabled and organized by Projekt DEAL. No funding was received to assist with the preparation of 
this manuscript.
Data availability
The implementation, experimental details, and obtained raw results are available through the repository linked in the 
Evaluation section.
Declarations
Ethics approval and consent to participate
Not applicable.
Competing interests
The authors declare no competing interests.
Received: 15 October 2024 Accepted: 2 April 2025
References
Accorsi R, Stocker T (2012) Discovering workflow changes with time-based trace clustering. Data-Driven Process 
Discovery and Analysis. Springer, Berlin, Heidelberg, pp 154–168
Adams JN, Pitsch C, Brockhoff T (2023) van der Aalst W An experimental evaluation of process concept drift detection. Proceedings of the VLDB Endowment, 16;8. https:// doi. org/ 10. 14778/ 35945 12. 35945 17
Adams JN, van Zelst SJ, Rose T, van der Aalst W (2023) Explainable concept drift in process mining. Inf Syst 
114:102177. https:// doi. org/ 10. 1016/j. is. 2023. 102177
Barbon Junior S, Tavares GM, Da Costa VGT, Ceravolo P , Damiani E (2018) A framework for human-in-the-loop monitoring of concept-drift detection in event log stream. In: Companion Proceedings of the The Web Conference 
(WWW ’18), ACM, New York, pp 319–326. https:// doi. org/ 10. 1145/ 31845 58. 31863 43
Bose RJC, Van Der Aalst W, Žliobaitė I, Pechenizkiy M (2013) Dealing with concept drifts in process mining. IEEE Trans 
Neural Netw Learn Syst 25(1):154–171. https:// doi. org/ 10. 1109/ TNNLS. 2013. 22783 13
Brockhoff T, Uysal MS, van der Aalst W (2020) Time-aware concept drift detection using the earth mover’s distance. 
In: 2nd International Conference on Process Mining (ICPM), Padua, Italy, October 4-9, 2020, IEEE, pp 33–40. 
https:// doi. org/ 10. 1109/ ICPM4 9681. 2020. 00016
Cao Y, Lin L, Di Y, Li X, Chen W (2024) MDD: process drift detection in event logs integrating multiple perspectives. In: 
2024 IEEE International Conference on Web Services (ICWS), pp 1125–1134. https:// doi. org/ 10. 1109/ ICWS6 2655. 
2024. 00135
Carmona J, Gavaldà R (2012) Online techniques for dealing with concept drift in process mining. In: Advances in 
Intelligent Data Analysis XI: IDA 2012, Springer, Berlin, Heidelberg, pp 90–102. https:// doi. org/ 10. 1007/ 978-3- 
642- 34156-4_ 10
Ceravolo P , Tavares GM, Junior SB, Damiani E (2022) Evaluation goals for online process mining: A concept drift per -
spective. IEEE Trans Serv Comput 15(4):2473–2489. https:// doi. org/ 10. 1109/ TSC. 2020. 30045 32
Ciccio CD, Mecella M (2015) On the discovery of declarative control flows for artful processes. ACM Trans Manag Inf 
Syst 5(4):1–37. https:// doi. org/ 10. 1145/ 26294 47
de Sousa RG, Peres SM, Fantinato M, Reijers HA (2021) Concept drift detection and localization in process mining: an 
integrated and efficient approach enabled by trace clustering. In: Proceedings of the 36th Annual ACM Sympo -
sium on Applied Computing, SAC ’21. ACM, New York, pp 364–373. https:// doi. org/ 10. 1145/ 34128 41. 34419 18
Elkhawaga G, Abuelkheir M, Barakat SI, Riad AM, Reichert M (2020) CONDA-PM: a systematic review and framework 
for concept drift analysis in process mining. Algorithms 13(7):161. https:// doi. org/ 10. 3390/ a1307 0161
Gallego-Fontenla V, Vidal JC, Lama M (2021) A conformance checking-based approach for sudden drift detection in 
business processes. IEEE Trans Serv Comput 16(1):13–26. https:// doi. org/ 10. 1109/ TSC. 2021. 31200 31
Grimm J, Kraus A, van der Aa H (2022) CDLG: A tool for the generation of event logs with concept drifts. In: International Conference on Business Process Management: BPM 2022 (Demos), CEUR-WS. 3216;92–96. https:// ceur- ws. 
org/ Vol- 3216/ paper_ 241. pdf
Guan W, Cao J, Gu Y, Qian S (2023) AIMED: An automatic and incremental approach for business process model repair 
under concept drift. Inf Syst 119:102285. https:// doi. org/ 10. 1016/j. is. 2023. 102285
Hanga KM, Kovalchuk Y, Gaber MM (2022) PgraphD*: Methods for drift detection and localisation using deep learning 
modelling of business processes. Entropy 24(7). https:// doi. org/ 10. 3390/ e2407 0910
Hassani M (2019) Concept drift detection of event streams using an adaptive window. In: 33rd International ECMS 
Conference on Modelling and Simulation: ECMS 2019, pp 230–239. https:// doi. org/ 10. 7148/ 2019- 0230

## Page 25 / 26

Page 25 of 26
Kraus and van der Aa Process Science (2025) 2:5 
 
Hompes B, Buijs JC, van der Aalst W, Dixit PM, Buurman J (2015) Detecting changes in process behavior using comparative case clustering. In: Data-Driven Process Discovery and Analysis: SIMPDA 2015, Springer, Cham, pp 54–75. 
https:// doi. org/ 10. 1007/ 978-3- 319- 53435-0_3
Huete J, Qahtan AA, Hassani M (2023) PrefixCDD: Effective online concept drift detection over event streams using 
prefix trees. In: 2023 IEEE 47th Annual Computers, Software, and Applications Conference (COMPSAC), pp 328–333. 
https:// doi. org/ 10. 1109/ COMPS AC577 00. 2023. 00051
Impedovo A, Mignone P , Loglisci C, Ceci M (2020) Simultaneous process drift detection and characterization with pattern-based change detectors. In: 23rd International Conference on Discovery Science: DS 2020, Thessaloniki, Greece, 
October 19–21, 2020, Proceedings 23, Springer, Cham, pp 451–467. https:// doi. org/ 10. 1007/ 978-3- 030- 61527-7_ 30
Klijn EL, Mannhardt F, Fahland D (2024) Multi-perspective concept drift detection: Including the actor perspective. In: 
Advanced Information Systems Engineering: CAiSE 2024, Springer, Cham, pp 141–157. https:// doi. org/ 10. 1007/ 
978-3- 031- 61057-8_9
Kraus A, van der Aa H (2024) Looking for change: A computer vision approach for concept drift detection in process mining. In: International Conference on Business Process Management, Springer, Cham, pp 273–290. https:// doi. org/ 10. 
1007/ 978-3- 031- 70396-6_ 16
Lakshmanan GT, Keyser PT, Duan S (2011) Detecting changes in a semi-structured business process through spectral 
graph analysis. In: IEEE 27th International Conference on Data Engineering Workshops, pp 255–260. https:// doi. org/ 
10. 1109/ ICDEW. 2011. 57676 40
Lin TY, Goyal P , Girshick R, He K, Dollár P (2017) Focal loss for dense object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 2980–2988
Lin TY, Maire M, Belongie S, Hays J, Perona P , Ramanan D, Dollár P , Zitnick CL (2014) Microsoft COCO: Common objects in 
context. In: Computer Vision: ECCV 2014, Springer, Cham, pp 740–755. https:// doi. org/ 10. 1007/ 978-3- 319- 10602-1_ 
48
Lin L, Wen L, Lin L, Pei J, Yang H (2020) LCDD: detecting business process drifts based on local completeness. IEEE Trans 
Serv Comput 15(4):2086–2099. https:// doi. org/ 10. 1109/ TSC. 2020. 30327 87
Liu L, Ouyang W, Wang X, Fieguth P , Chen J, Liu X, Pietikäinen M (2020) Deep learning for generic object detection: A 
survey. Int J Comput Vis 128:261–318. https:// doi. org/ 10. 1007/ s11263- 019- 01247-4
Liu N, Huang J, Cui L (2018) A framework for online process concept drift detection from event streams. In: 2018 IEEE 
International Conference on Services Computing (SCC), pp 105–112. https:// doi. org/ 10. 1109/ SCC. 2018. 00021
Lu Y, Chen Q, Poon SK (2021) A robust and accurate approach to detect process drifts from event streams. In: International Conference on Business Process Management, Springer International Publishing, Cham, pp 383–399. https:// 
doi. org/ 10. 1007/ 978-3- 030- 85469-0_ 24
Luengo D, Sepúlveda M (2012) Applying clustering in process mining to find different versions of a business process that 
changes over time. In: Business Process Management Workshops: BPM 2011 Workshops, Clermont-Ferrand, France, 
August 29, 2011, Revised Selected Papers, Part I 9, Springer, Berlin, Heidelberg, vol 99. pp 153–158. https:// doi. org/ 
10. 1007/ 978-3- 642- 28108-2_ 15
Maaradji A, Dumas M, La Rosa M, Ostovar A (2017) Detecting sudden and gradual drifts in business processes from 
execution traces. IEEE Trans Knowl Data Eng 29(10):2140–2154. https:// doi. org/ 10. 1109/ TKDE. 2017. 27206 01
Maaradji A, Dumas M, La Rosa M, Ostovar A (2015) Fast and accurate business process drift detection. In: 13th International Conference on Business Process Management: BPM 2015, Innsbruck, Austria, August 31 – September 3, 2015, 
Proceedings 13, Springer, Cham, vol 9253. pp 406–422. https:// doi. org/ 10. 1007/ 978-3- 319- 23063-4_ 27
Manoj Kumar M, Thomas L, Annappa B (2015) Capturing the sudden concept drift in process mining. Algorithms & Theories for the Analysis of Event Data ATAED’15, June 22-23, vol 1371. Brussels, Belgium, pp 132–134
Martjushev J, Bose RJC, Van Der Aalst W (2015) Change point detection and dealing with gradual and multi-order 
dynamics in process mining. In: Perspectives in Business Informatics Research. BIR 2015, Springer, Cham, pp 
161–178. https:// doi. org/ 10. 1007/ 978-3- 319- 21915-8_ 11
Meira Neto AC, de Sousa RG, Fantinato M, Peres SM (2024) Revisiting the transition matrix-based concept drift approach: 
Improving the detection task reliability through additional experimentation. SN Comput Sci 5(1):188. https:// doi. 
org/ 10. 1007/ s42979- 023- 02536-z
Neto ACM, de Sousa RG, Fantinato M, Peres SM (2023) Towards a transition matrix-based concept drift approach: 
Experiments on the detection task. In: Proceedings of the 25th International Conference on Enterprise Information 
Systems - Volume 2: ICEIS, INSTICC, SciTePress, pp 361–372. https:// doi. org/ 10. 5220/ 00118 43600 003467
Nguyen H, Dumas M, La Rosa M, ter Hofstede AH (2018) Multi-perspective comparison of business process variants 
based on event logs. In: Conceptual Modeling. ER 2018, Springer, Cham, pp 449–459. https:// doi. org/ 10. 1007/ 978-3- 
030- 00847-5_ 32
Ostovar A, Leemans SJ, Rosa ML (2020) Robust drift characterization from event streams of business processes. ACM 
Trans Knowl Discov Data 14(3):1–57. https:// doi. org/ 10. 1145/ 33753 98
Ostovar A, Maaradji A, La Rosa M, ter Hofstede AH (2017) Characterizing drift from event streams of business processes. 
In: Advanced Information Systems Engineering: 29th International Conference, CAiSE 2017, Essen, Germany, June 
12-16, 2017, Proceedings 29, Springer, Cham, pp 210–228. https:// doi. org/ 10. 1007/ 978-3- 319- 59536-8_ 14
Ostovar A, Maaradji A, La Rosa M, ter Hofstede AHM, van Dongen BFV (2016) Detecting drift from event streams of 
unpredictable business processes. In: Conceptual Modeling. ER 2016, Springer International Publishing, Cham, pp 
330–346. https:// doi. org/ 10. 1007/ 978-3- 319- 46397-1_ 26
Pasquadibisceglie V, Appice A, Castellano G, Malerba D, Modugno G (2020) Orange: outcome-oriented predictive process 
monitoring based on image encoding and cnns. IEEE Access 8:184073–184086. https:// doi. org/ 10. 1109/ access. 
2020. 30293 23
Pfeiffer P , Lahann J, Fettke P (2021) Multivariate business process representation learning utilizing gramian angular fields 
and convolutional neural networks. In: International Conference on Business Process Management, Springer International Publishing, Cham, pp 327–344. https:// doi. org/ 10. 1007/ 978-3- 030- 85469-0_ 21
Qahtan AA, Alharbi B, Wang S, Zhang X (2015) A PCA-based change detection framework for multidimensional data 
streams: Change detection in multidimensional data streams. In: Proceedings of the 21th ACM SIGKDD International

## Page 26 / 26

Page 26 of 26Kraus and van der Aa Process Science (2025) 2:5 
Conference on Knowledge Discovery and Data Mining, KDD ’15. ACM, New York, pp 935–944. https:// doi. org/ 10. 
1145/ 27832 58. 27833 59
Richter F, Seidl T (2017) Tesseract: time-drifts in event streams using series of evolving rolling averages of completion 
times. In: 15th International Conference on Business Process Management: BPM 2017, Barcelona, Spain, September 
10–15, 2017, Proceedings 15, Springer International Publishing, Cham, pp 289–305. https:// doi. org/ 10. 1007/ 978-3- 
319- 65000-5_ 17
Sato DMV, Barddal JP , Scalabrin EE (2021) Interactive process drift detection framework. In: International Conference on 
Artificial Intelligence and Soft Computing, Springer, Springer International Publishing, Cham, pp 192–204. https:// 
doi. org/ 10. 1007/ 978-3- 030- 87897-9_ 18
Sato DMV, De Freitas SC, Barddal JP , Scalabrin EE (2021) A survey on concept drift in process mining. ACM Comput Surv 
54(9):1–38. https:// doi. org/ 10. 1145/ 34727 52
Seeliger A, Nolle T, Mühlhäuser M (2017) Detecting concept drift in processes using graph metrics on process graphs. In: 
Proceedings of the 9th Conference on Subject-Oriented Business Process Management, S-BPM ONE ’17, vol 9. ACM, 
New York, pp 1–10. https:// doi. org/ 10. 1145/ 30405 65. 30405 66
Smirnov S, Weidlich M, Mendling J (2012) Business process model abstraction based on synthesis from well-structured 
behavioral profiles. Int J Coop Inf Syst 21(01):55–83. https:// doi. org/ 10. 1142/ S0218 84301 24000 35
Stertz F, Rinderle-Ma S (2018) Process histories-detecting and representing concept drifts based on event streams. In: On 
the Move to Meaningful Internet Systems: OTM 2018 Conferences, Valletta, Malta, October 22-26, 2018, Proceedings, 
Part I, Springer, pp 318–335. https:// doi. org/ 10. 1007/ 978-3- 030- 02610-3_ 18
Stertz F, Rinderle-Ma S (2019) Detecting and identifying data drifts in process event streams based on process histories. 
In: Information Systems Engineering in Responsible Information Systems: CAiSE Forum 2019, Rome, Italy, June 3–7, 
2019, Proceedings 31, Springer, pp 240–252. https:// doi. org/ 10. 1007/ 978-3- 030- 21297-1_ 21
Tavares GM, Ceravolo P , Turrisi Da Costa VG, Damiani E, Barbon Junior S (2019) Overlapping analytic stages in online 
process mining. In: 2019 IEEE International Conference on Services Computing, pp 167–175. https:// doi. org/ 10. 
1109/ SCC. 2019. 00037
van der Aalst W (2016) Process Mining: Data Science in Action. Springer, Berlin, Heidelberg. https:// doi. org/ 10. 1007/ 
978-3- 662- 49851-4_1
van Der Aalst W, Pesic M (2009) Schonenberg H (2009) Declarative workflows: Balancing between flexibility and support. 
Comput Sci Res Dev 23:99–113. https:// doi. org/ 10. 1007/ s00450- 009- 0057-9
Van Der Aalst W, et al (2011) Process mining manifesto. In: Business Process Management Conference. BPM Workshop, 
Springer, pp 169–194. https:// doi. org/ 10. 1007/ 978-3- 642- 28108-2_ 19
Weber P , Bordbar B, Tino P (2011) Real-time detection of process change using process mining. In: Imperial College 
Computing Student Workshop, vol DTR11-9. Citeseer, Imperial College London, pp 108–114
Xu J, Zhang Y, Duan Q (2023) Concept drift detection and localization framework based on behavior replacement. Appl 
Intell 53(13):16,776–16,796. https:// doi. org/ 10. 1007/ s10489- 022- 04341-2
Yang L, McClean S, Donnelly M, Burke K, Khan K (2021) Process duration modelling and concept drift detection for business process mining. In: IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, 
Scalable Computing & Communications, Internet of People and Smart City Innovation, pp 653–658. https:// doi. org/ 
10. 1109/ SWC50 871. 2021. 00097
Yeshchenko A, Di Ciccio C, Mendling J, Polyvyanyy A (2021) Visual drift detection for event sequence data of business 
processes. IEEE Trans Vis Comput Graph 28(8):3050–3068. https:// doi. org/ 10. 1109/ TVCG. 2021. 30500 71
Zellner L, Richter F, Sontheim J, Maldonado A, Seidl T (2021) Concept drift detection on streaming data with dynamic 
outlier aggregation. In: Process Mining Workshops. ICPM 2020, Springer International Publishing, Cham, pp 206–217. 
https:// doi. org/ 10. 1007/ 978-3- 030- 72693-5_ 16
Zheng C, Wen L, Wang J (2017) Detecting process concept drifts from event logs. In: On the Move to Meaningful Internet 
Systems: OTM 2017 Conferences, Springer International Publishing, Cham, pp 524–542. https:// doi. org/ 10. 1007/ 
978-3- 319- 69462-7_ 33
Publisher’s Note
Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
