# ConceptExplorer: Visual Analysis of Concept Drift in Multi-Source Time-Series Data

**Authors:** Wang et al.

**Venue:** IEEE VAST (2020), arXiv version — Visual analytics prior art

*Source PDF: `14_Wang-2020-ConceptExplorer-VAST-arXiv.pdf`*

*Converted to markdown following the Selected_Papers strategy (full-text extraction, page-ordered). Verify title/authors/year against publisher before citing.*

---

## Page 1 / 12

arXiv:2007.15272v2 [cs.HC] 18 Aug 2020
ConceptExplorer: Visual Analysis of Concept Drifts
in Multi-source Time-series Data
Xumeng Wang*
State Key Lab of CAD&CG
Zhejiang University
Wei Chen†
State Key Lab of CAD&CG
Zhejiang University
Jiazhi Xia‡
School of Computer Science
and Engineering
Central South University
Zexian Chen§
State Key Lab of CAD&CG
Zhejiang University
Dongshi Xu¶
College of Information
Engineering
China Jiliang University
Xiangyang Wu||
Institute of Graphics and
Image
Hangzhou Dianzi University
Mingliang Xu**
Zhengzhou University
T obias Schreck††
Graz University of T echnology
(a)
(b) 
(c) 
(d)(e) 
Figure 1: ConceptExplorer contains ﬁve main views. (a) The data entrance introduces th e applied data sources and attributes. (b)
The timeline navigator view is used to select interested tim e segments; (c) The prediction model view presents the train ing process
of prediction models to explain concept drift detection mod el; (d) The concept-time view shows the time segments recomm ended
for analyzing concepts based on the moment selected by analy sts in prediction model view. (e) The explanation view compa res
concepts pairwise through a correlation matrix.
ABSTRACT
Time-series data is widely studied in various scenarios, like weather
forecast, stock market, customer behavior analysis. To com prehensively learn about the dynamic environments, it is necessary to comprehend features from multiple data sources. This paper pro poses a
*e-mail: wangxumeng@zju.edu.cn
†e-mail: chenwei@cad.zju.edu.cn
‡e-mail: xiajiazhi@csu.edu.cn
§e-mail: zexianchen@zju.edu.cn
¶e-mail: p1703085223@stu.cjlu.edu.cn
||e-mail: wuxy@hdu.edu.cn
**e-mail: iexumingliang@zzu.edu.cn
††e-mail: tobias.schreck@cgv.tugraz.at
Wei Chen and Jiazhi Xia are corresponding authors.
novel visual analysis approach for detecting and analyzing concept
drifts from multi-sourced time-series. We propose a visual detection scheme for discovering concept drifts from multiple so urced
time-series based on prediction models. We design a drift le vel index to depict the dynamics, and a consistency judgment model to
justify whether the concept drifts from various sources are consistent. Our integrated visual interface, ConceptExplorer, facilitates
visual exploration, extraction, understanding, and compa rison of
concepts and concept drifts from multi-source time-series data. We
conduct three case studies and expert interviews to verify t he effectiveness of our approach.
Index Terms: Temporal data; data analysis, reasoning, problem
solving, and decision making; machine learning techniques .
1 I NTRODUCTION
Facing the changing world, analysts track the underlying re lationship between the interested target and the environment to un derstand, explain, and predict the evolving events. We use the t erm

## Page 2 / 12

concept [18] to describe the underlying relationship between the
interested target variable and the environment variables i n timeseries data. The concepts evolve and are diverse across diff erent
data sources, e.g., data from different groups or regions. We denote the change of concept as concept drift [38]. Tracking the
concept drift is of theoretical and practical signiﬁcance f or domain
experts, like portfolio selection [25]. It provides the kno wledge of
dynamical concepts, updates the understanding of underlyi ng relationships, and sharpens the insights into the difference s among
various groups.
For example, ﬁnancial experts are interested in the price ﬂu ctuation of stocks. They would like to track the relationship be tween
the stock index ( i.e., the target variable) and various economic indicators (i.e., the environment variables). By analyzing the concepts
in a period, they can identify the most correlated economic i ndicators. This knowledge is helpful to explain the factors and predict the
trend of stock indices. When concept drift is identiﬁed, the ir understanding of underlying relationships is updated with the di sclosed
change. The concept drifts in different stock markets are re lated
and, however, heterogeneous [6]. For instance, the ﬂuctuat ion of
stock indices of the U.S. market would be different but relat ed to
that of the European market. The knowledge of and comparison between the concept drifts in different markets sharpens the i nsights
into the relationship and difference among multiples stock markets.
Another example is to inspect the cure rate during a virus spr eading.
The target variable is the cure rate and the environment vari ables
include the health proﬁle of patients. Knowing the concept d rifts
in different regions is important for experts to custom the d iagnosis
strategies.
Tracking concept drifts from the huge number of multi-sourc e
time-series data is technically demanding. There are two ma jor
challenges to be addressed. The ﬁrst one is to model concept d rifts
and identify them in the time-series data [22, 40]. To detect certain patterns in multivariate time-series data, domain exp erts need
to manually set the patterns and parameters. However, exper ts have
not an explicit description of the underlying relationship . The recent time-series clustering [24, 31, 36] and frequency-bas ed detection [29] are usually used to detect pattern automatically w ithout
specifying the target ﬁrst. Because they assume that the sta te of a
process is repetitive, these approaches are suitable for pe riodic processes. However, the concept drift occurrences in real-world scenarios are always evolving irregularly. The cluster-based or f requencybased approaches cannot achieve the desired performance. T he second challenge is to design an intuitive visualization to ill ustrate the
evolving concept drifts over time and the relationship amon g concept drifts across multiple data sources [41]. The concept d rifts
often occur hundreds of times in a rapidly changed period. Th e
data heterogeneity among data sources results in complicat ed relationships among concept drifts. Therefore, an efﬁcient int eractive
visualization is necessary to present the concept drifts ov er a long
time span, which allows experts to focus on a selected time pe riod
in interest for further inspection.
We have developed a visual analytics system, ConceptExplorer,
to address these challenges. To model the concept, we have em -
ployed prediction models to capture the underlying relatio nship between the target variable and the environment variables. Th e concept drift is captured by the accuracy change of prediction m odels.
Based on the performance of prediction models, we have formu -
lated the drift level index to indicate concept drifts and pr oposed
a consistency judgment model to inspect the inconsistency a mong
multi-sources. To effectively convey complicated conceptdrifts and
their relationships, we have developed an interactive visu alization
that combines the strengths of multiple views. In particula r, we
have developed a timeline navigator view to help users quick ly get
an overview of occurred concept drifts in a long time span in m ultiple sources. The prediction model view presents the featu re of
prediction models and thus sharpen the insights into concep t drifts.
The concept-time view allows analysts to focus on and ﬁne-tu rn the
period of a certain concept. The concept explanation view em ploys
a matrix visualization to present the correlation between t he target
variable and environment variables. It supports the compar ison between two concepts by diagonally juxtaposing them in the mat rix.
These visualizations are coordinated to support the intera ctive analysis of concept drifts from multi-source time-series data. Lastly, we
conduct three case studies with real application scenarios and collect feedback from three experts to verify the effectivenes s of our
approach.
The major contributions of this work are:
• A visual analytics system that helps experts understand an d
analyze the concept drifts over time and their relationship s
across multiple data sources.
• A description of the concept drift that takes advantage of p rediction models; it induces a set of models for concept drift
detection and comparison.
• A coordinated visualization that combines a set of novel de -
signs in the timeline visualization, matrix visualization , and
prediction model view.
2 R ELATED WORK
We surveyed existing studies from two aspects: 1) detection of concept drifts, and 2) visualization of time-series data.
2.1 Detection of Concept Drifts
Concept drifts are deﬁned as the changes in the joint distrib ution
between the environment variables ( i.e., the time-series data that
analysts collected) and target variables( i.e., the labels that analysts want to predict) [18, 38]. Approaches for detecting co ncept
drifts are fall in two categories: performance-based appro aches and
distribution-based approaches. Performance-based appro aches detect concept drifts from the abnormal ﬂuctuation of perform ance
indicators, like accuracy. Drift Detection Method (DDM) [1 7] recognizes an abnormal increase in error rate over certain rang es as a
warning or a concept drift occurrence. The ranges are determ ined
by the conﬁdence intervals of the Normal distribution. A var iety of
statistical test methods can be applied after the study subj ect transforms into the error rate. For instance, concept drifts are i dentiﬁed
by a set of the chi-square test [35]. To avoid the inﬂuence of t he
size of the upcoming data, Fishers Exact test is chosen [11].
Following the deﬁnition of the concept, distribution-base d approaches compare data distributions and detect concept dri fts by
identifying distribution changes. However, calculating d istribution
similarity is a time-consuming task due to the complexity of distribution characteristics, i.e., extreme values, skewness, variance,
etc. Considering an incremental learning process, the prob lem can
be simpliﬁed by focusing on the differences. Certain assump tions
are made [15] to describe the changes as a series of operation s
between constant values. Besides, grouping variables is a p rominent approach to simplify the density statistics process. T he framework [39] maps variables into a grid space and performs densi tybased clustering on grid cells. Then the inﬂuences caused by the
upcoming data can be summarized as the cluster generations o r
cluster extensions. Taking advantage of k-nearest neighbor (KNN),
sub-spaces can be constructed [28] for a sample set, and dens ity
variations are identiﬁed with a distance measurement. In su mmary, distribution-based approaches need to be supported b y intricate quantitative evaluation [32].
2.2 Visualization of Time-series Data
Identifying patterns from multidimensional time-series d ata is a
comprehensive process. It is thus essential to integrate vi sualization and data analysis methods for better efﬁciency. Existi ng studies achieve this goal from three perspectives: setting targ et patterns

## Page 3 / 12

interactively, extracting repeated patterns based on clus tering and
frequency features, and detect abnormal patterns by levera ging machine learning approaches.
Within a visual interface, analysts can express what they wa nt
from the visual analysis system. Thermalplot [43] supports specify by setting weights for each attribute. Time-varying obj ects are
mapped into a two-dimensional space deﬁned by the degree of interest and corresponding change over time. To explore co-occur rence
patterns, COPE [26] needs analysts to specify events by sett ing
thresholds for attribute values. The spatiotemporal patte rn of similar events can be checked with COPE. It is effective to allow u sers
to gradually narrow their search, especially when they are n ot sure
what they want. TimeNotes [46] provide users with a hierarch y
time axes. Users are allowed to iteratively select one or mor e small
time range of interest from a large time range by brushing.
When analysts have limited knowledge of datasets, it is necessary to augment the analysis with automatic methods. Temp oral Multidimensional Scaling (TMDS) [21] discretizes the t ime dimension by a user-deﬁned sliding window, and projects the mu ltidimensional data in each window to one dimension via MDS. Aft er
a series of ﬂipping operations, one-dimensional projectio ns are juxtaposed to show temporal patterns. In addition to dimension ality reduction methods, extracting important periods can also red uce the
user’s workload. StreamExplorer [51] employs a subevent de tection model to identify important periods from a social strea m. Because major events always lead to popular discussions, Stre amExplorer recommends users the periods with a large number of tw eets
to analyze related events. To extract patterns ﬂexibly, TPF low [29]
employs a piecewise rank-one tensor decomposition to detec t subtensors (i.e. multidimensional patterns) with a top priority.
If unpredictable events are regarded as abnormal, the high prediction error of automatic models may imply abnormal patterns [ 45].
Taking advantage of the same feature, concept drift detecti on can
be applied to locate useful patterns from dynamic environments. Visualization techniques have been used to depict the develop ment of
concept drifts [12,53]. Common charts, like line charts [49 ], scatter
plots [42] and parallel coordinates [37] are employed for th is purpose. On the other hand, model-generating information can c onvey
the characteristics of concept drifts. The time-varying co ntributions
of each attribute value can be visualized for classiﬁcation [13]. By
marking the ﬂuctuations, the occurrences of concept drifts can be
easily identiﬁed from a micro-level. Also, users are allowe d to take
an overview from a macro level to assess the importance of eac h attribute value [13]. None of these studies can explain why a co ncept
drift is identiﬁed by the detection model, which is viral for the ﬁnal
decision.
3 P ROBLEM DEFINITION AND MODELS
Two models are applied to support our goals. Before introduc ing
the goals and the models, we ﬁrst explain related deﬁnitions .
3.1 Deﬁnitions
In this work, time-series from multiple sources are present ed as
temporal data records . Each data record contains multiple environment variables X and a target variable y. The data records
are distributed non-uniformly along the timeline. We group data
records in a unit time segment into a batch. A batch forms a basic unit for coordinated analysis among different data sour ces. We
denote the data source and timestamp of a batch as its context.
In machine learning scenarios, records are usually used by t he
training model, where the target variable is the data label. Without
loss of generality, the target variable is limited to be binary in this
paper. It is assumed that there is an underlying relationship between
the environment variables and the target variable, e.g., an underlying mapping y = f (X), or a conditional distribution p(y |X) [18]. A
concept refers to such a relationship. The changes of concept over
time is denoted as concept drift [50].
3.2 Goals
Our main goal is a visual analytics tool for identifying and u nderstanding concept drifts in multiple time-series data. We id entify
four goals in building such a visual analytics system:
G1: Automatic identiﬁcation of concept drifts and concepts .
It is laborious to browse the time-series data through the en tire time
span. Moreover, the concept and concept drift are only impli citly
embedded in the time-series data. Identifying numerous con cepts
and concept drifts individually is cumbersome. Therefore, the ﬁrst
goal of our system is to support the automatic identiﬁcation of stable concept and important concept drifts.
G2: Visual representation of concepts. There is not an explicit
deﬁnition of concepts. The assumed deﬁnitions, including t he mapping or the conditional probability, are complicated to des cribe and
understand. Therefore, presenting visualization to discl ose the pattern of concepts, i.e., the relationship between the target variable
and environment variables is needed.
G3: Discrimination of concept drifts in multiple data sources.
With the same target variable and environment variables are collected, concepts drifts could be heterogeneous in differen t data
sources [20]. When an inconsistency occurs among different data
sources, analysts should be able to discriminate them and ve rify the
interested concept drift.
G4: Interactive speciﬁcation of concepts. Concepts extracted
from the batches with different contexts may be numerous. Al -
though automatic models can augment the selection process, there
is a natural need for interactive exploration and speciﬁcat ions of
concepts [16, 33].
3.3 The Drift Level Index
Conventional machine learning approaches provide a quanti tative
description of concept drifts ( G1). The basic idea is that if the concept is stationary over time, the performance of a trained pr ediction
model should be stable or increasing. Otherwise, the predic tion accuracy will decrease and trigger an index of concept drifts. Therefore, the decrease of the prediction accuracy is a meaningfu l index
for concept drifts.
When a concept drift occurs, the pre-trained prediction mod el
might have a decreasing performance and be less sensitive to subsequent concept drifts. Therefore, the prediction model shou ld update
with the evolving of concepts, and hence the learning process is iterative, i.e., learning parameters for each attribute are updated when
a new data record comes. Following the technique presented i n [8],
we maintain a set of prediction models and take the one with th e
highest accuracy in the last veriﬁcation as the output model . The
weakest model in the set is replaced with a newly trained mode l.
Models trained on the new data can learn new characteristics of the
label and be adaptable to dynamic environments. As a result, the
prediction process has a high performance and is sensitive t o the
upcoming concept drifts.
We use pi to denote the error rate of the prediction model in
a sliding window, which ends at the ith record. The sliding window is set to cover 500 data records in our implementation. Th e
distribution of correct predictions in a sliding window can be regarded as a binomial distribution. Therefore, for each pi, we compute the standard deviation as si =
√
pi(1 − pi)/n. Similar to the
work in [17], we assume that the error rates are distributed normally.
The concept drift level can be measured by the conﬁdence leve ls of
corresponding conﬁdence intervals. As discussed above, in a static
environment, the error rate of a prediction model is suppose d to be
decreasing or approximately stable over time. Although the ﬂuctuation of the error rate is normal, the degree of its increasing typically
indicates a high probability of the occurrence of a concept d rift. As

## Page 4 / 12

shown in Figure 2, the probability of a concept drift is compu ted
with the minimum of the error rates pmin (after the latest concept
drift) and the standard deviation smin [17]. With the veriﬁed record
i, the drift level ri is deﬁned as:
ri = pi + si − pmin
smin
(1)
The threshold to determine whether a concept drift appears, i.e., the
conﬁrmation level, is set to be ri ≥ 3, which implies that the conﬁdence level of a concept drift occurrence exceeds 99% [17]. A lso,
when ri ≥ 2 ( i.e., the warning level, the corresponding conﬁdence
level is over 95%), a warning is issued. The drift level rt for a
batch is regarded as the average of ri, ri+1, ...,r j, where i, i + 1, ..., j
denote the output sequence from the data records in the batch .
Error ratepmin
#Occurrence
smin smin
smin
pmin
The confidence interval for 99%The confidence interval for 95%
! The warning level
The confirmation level
If a pi + si is in , a concept drift is detected
, a warning is issued
{
Figure 2: Explanation of the drift detection model [17].
3.4 The Consistency Judgment Model
The similarity between concepts derived from different data sources
can be represented by the parameter similarity of the predic tion
models. However, parameters of the prediction models can no t
summarize the consistency of concept drifts. To support G3, we
need to learn about the response of each data source to the dyn amic
environment. The dynamic environment can be described by th e
time-varying drift levels of data sources, which are contin uously
captured with the training of prediction models. Supposed t hat a
data source has a consistent response with others, the Na ´
’ive Bayes
theory is employed to infer the time segment of the concept dr ift occurrence. The judgment about whether the detected concept d rifts
satisfy the inferred results can be concluded.
Considering the differences among data sources, we propose
a consistency judgment model that is used for each data sourc e
separately. At time t, the drift levels of m data sources
d1, d2, ...,di, ...,dm are recorded as rt
1, rt
2, ...,rt
i , ...,rtm. Regarding
xt
i = [rt
1, ...,rt
i− 1, rt
i+1, ...,rtm] as inputs of the judgment model, the
corresponding label yt
i of data source di, is deﬁned as whether a
concept drift occurs in the recent time segment ∆t (normally as a
unit time segment), that is, whether the conﬁrmation level e xceeds
by one of rt− ∆t
i , rt− ∆t+1
i , ...,rt+∆t
i . Based on the Na¨ ıve Bayes theory, a probability curve can be generated to depict the proba bilities
of concept drifts over time. The curve segments, where the pr obability is higher than a user-deﬁned probability threshold c, implies
the corresponding labels are judged as “yes”. On the contrar y, the
labels of the rest time segments are “no”. Therefore, the tim e segment [tstart ,tend] for an occurrence of a concept drift is inferred. If
the label yi is independent with xi, the corresponding probability
curve would be ﬂat and has less chance to be higher than the thr eshold. That is, if the data source i is inconsistent, the consistency
judgment model results in zero time segment.
We compare the time points of detected concept drifts with th e
time segments to verify if a data source drifts in an inconsis tent
way. When the label is deﬁned, the corresponding concept dri fts
should appear in the time segment [tstart − ∆t,tend + ∆t]. A data
source is regarded as inconsistent with the entire environm ent at t if
its previous concept drift occurs out of the previous time se gment.
Thus, the larger the parameter c is, the higher the probability that
data sources are considered as inconsistent.
4 S YSTEM OVERVIEW
4.1 Design Requirements
To address the goals mentioned in Section 3.2, ﬁve design req uirements are identiﬁed to guide system design.
DR1: Provide an overview of concept drift occurrences over
time. The drift occurrences over time can help analysts identify
the interesting time segment. When the analysts’ preferenc es are
unknown, interactive and hierarchical exploration of occu rrences
along time intervals is needed [34, 46].
DR2: Integrate features of concept drifts from the prediction models. Concept drifts hinder existing prediction models (the
model trained from historical data) from accurate predictions in new
environments. The accuracy ﬂuctuation of the prediction mo del indicates the occurrence of concept drifts [7, 30, 42, 52, 54]. In addition, parameters of prediction models can reﬂect the rela tionship
between different inputs and the label, that is, the model’s understanding of the concept [9].
DR3: Identify the context of concepts and allow adjustments .
Analysts need to know the context of the analyzed data record s.
Considering that analysts may miss details or may disagree w ith
the navigation, interactive adjustments for recommended results are
needed [23, 27].
DR4: Study the relationship between attributes and labels.
While concepts have not an explicit deﬁnition, it is essenti al to
provide a visual explanation. The relationship between lab els and
attributes are considered to be an important description of concepts [14, 18].
DR5: Compare concepts in different contexts. Comparing
different concepts facilitates the understanding of the ev olving concepts and their contexts, e.g., the trends and outliers. There is a
need to compare a newly identiﬁed concept with previously st udied
ones and record the identiﬁed concepts [49]. Comparing conc epts
related to a concept drift also favors the understanding of t he drift.
4.2 Workﬂow
To derive concepts from multi-source time-series datasets , analysts
need to manipulate the data and select contexts. We design a ﬁ vestep workﬂow, as shown in Figure 3.
We provide an overview of all concept drifts detected from al l
data sources during the entire time span (see Figure 3(a), DR1).
Analysts may be attracted by time segments when a single data
source has interesting patterns ( e.g. dense occurrences or a periodicity), or multiple data sources need to be compared ( e.g. abnormal concept drifts). When a time segment is selected, con cept
drifts from the prediction models are shown ( DR2). As shown in
Figure 3(b), the accuracy ﬂuctuation and parameters of pred iction
models can assist the detection of concept drifts. Next, ana lysts can
specify the context of the concept to be analyzed with the ext ernal
knowledge of concept drifts, as shown in Figure 3(c). ConceptExplorer can recommend the time segment between two adjacent concept drifts according to an analyst-speciﬁed time point. Th e recommended time segments for different data sources may be diffe rent,
or even inconsistent. ConceptExplorer assesses the consistency of
data sources by the consistency judgment model and recommen ds
the group of data sources with consistent concept drifts. Th e recommended selection is displayed ( DR3). If analysts are not satisﬁed
with the recommendations, they can make ﬂexible adjustment s on

## Page 5 / 12

contexts to support special analysis tasks. To explore the c oncepts
with speciﬁed contexts, the relationship between attribut es and concepts ( DR4, see Figure 3(d)) are visualized. Analysts can identify
and record signiﬁcant concepts that may be involved in subse quent
analysis. The identiﬁed concepts can be compared with other concepts (DR5, see Figure 3(e)).
(a) Concept drift
overview
(d) Concepts analysis
and comparison
(b) Concept drift
inspection
Accuracy fluctuations
Parameter changes
Select time intervals
(c) Context specification
Select data sources
(e) Identified concepts
Abnormal concepts
Figure 3: The ﬁve-step workﬂow: (a) Observing the distribut ion of
concept drifts and warnings; (b) Inspecting concept drifts through accuracy ﬂuctuation and parameter changes; (c) Specifying th e context
of the concept to be analyzed; (d) Analyzing and comparing co ncepts
based on correlations; (e) Identifying interesting concep ts.
5 C ONCEPT EXPLORER
As shown in Figure 1, ConceptExplorer consists of a data entrance
(see Figure 1(a)) and four views. The data entrance lists the label
deﬁnition, description of data sources, and attributes. Th e online
system is available through the link: http://101.132.126.253/.
5.1 The Timeline Navigator View
As required by DR1, the timeline navigator view presents the entire timeline and the indices of concept drifts from multipl e data
sources (see Figure 1(b)). Each row corresponds to a data sou rce.
ConceptExplorer assigns a unique color to each data source. Due to
the limited horizontal space, the distribution of concept d rifts may
be dense. ConceptExplorer employs a “ × ” to mark a concept drift,
which can highlight the speciﬁc moment by its intersection. Time
segments, in which the drift level exceeds a certain value (i nitialized as the warning level, namely, 2) are highlighted by “ − ”. These
marks indicate various patterns along the timeline, like de nse occurrences, outliers, inconsistency with other data sources, p eriodicity.
5.2 The Prediction Model View
The prediction model view (Figure 1(c)) supports DR2.
5.2.1 The Accuracy Fluctuation Chart
The line charts on the left (Figure 1(c)) show the accuracy ﬂu ctuation of the prediction models trained by the data from each d ata
source. The occurrences of concept drifts are labeled by “× ”, which
is the same as that in the timeline navigator view. In additio n, the
moments with warnings are encoded by hollow dots. To explain
concept drift detection, the accuracy ﬂuctuation chart vis ualizes the
magnitude of the accuracy drop of the time segments whose dri ft
levels are above the warning level (Figure 4(a)). Different data
sources may issue drift warnings at similar time segments. T o avoid
misunderstandings caused by overlaps, shifted stripes are employed
to highlight warning time segments (Figure 4(b)). It can be s een
that even when the warning segments of different data source s are
staggered, the start and end moments of different time segme nts
can be clearly distinguished. V ertical stripes are used bec ause they
can emphasize the height, that is, the magnitude of accuracy drops.
The results from the consistency judgment model are also sho wn in
the accuracy ﬂuctuation chart. If a concept drift is detecte d during
a time segment that is not included by the result from the cons istency judgment model, we emphasize them by a triangle mark to
distinguish from circles representing others (Figure 4(c) ).
Accuracy curve (1 - pi)
1 - p min
pi - pmin
(a) Magnitude of accuracy drops (b) Shifted stripes
(c) Glyph design
The interval generated by the 
consistency judgment model 
(Inconsistent) (Consistent) Warning: Concept drifts:
Figure 4: The visual designs for explaining the detection mo del and
the consistency judgment model. (a) The explanation corres ponds to
the formula mentioned in Section 3.3. (b) Strips are shifted to avoid
overlaps. (c) Encodings of concept drifts and warnings.
5.2.2 The Projected Parameter View
The model parameters updated after each batch during the ent ire
training process are projected into a two-dimensional plan e using
principal components analysis (PCA) based on singular valu e decomposition (SVD) [47]. The points projected by parameters of
the same data source are connected in order to form a curve. Th e
distance between each pair of projected points illustrates the similarity of corresponding model parameters, namely, the conc ept similarity described by prediction models. Evolution pattern s, like intersections and bundles [5, 19] can be identiﬁed from the for med
curves [10]. The convergence and dispersal of curve segment s
representing different data sources indicate the agreemen t and disagreement of related models on the understanding of the conc ept.
The curve segments corresponding to time segments selected in the
timeline navigator view is colored by transparency, from wh ich analysts can learn about the temporal order. The view is automa tically
zoomed in or out so as to ﬁt the curves of the entire training pr ocess
or the selected time segment within the window, as shown in Fi gure 5. With the background of the entire trajectories (Figur e 5(a)),
analysts can better measure relative distances. After zoom ing in, it
can be seen (Figure 5(b)) that curves are not overlapped but w ith
similar directions, that is, data sources have similar drif ts. Analysts
can drag the handle on the time axis of the accuracy ﬂuctuatio n
chart to move the circles, which highlight the projected par ameters
corresponding to the same moment.
5.3 The Concept-Time View
The concept-time view displays the time segments in differe nt data
sources that are integrated for concept analysis, as shown i n Figure 1(d). To display the sources of the applied data records, as

## Page 6 / 12

ča) The overview čb) The enlarged part
Figure 5: The parameter projections of prediction models ru nning
for all data sources. The opacity encodes the time order. (a) The
overview of the entire time range. (b) An enlarged part.
mentioned in DR3, each data source is listed in a row to distinguish
different data sources. Then, their data records are divide d individually to introduce a speciﬁc time. Analysts need to make t he
trade-off between the number of data records and the clarity of concepts for appropriate adjustments. To facilitate decision -making,
the drift level and the size of each batch are encoded with the color
and height of the bar, respectively. The batches that compos e the
data records to be analyzed are highlighted. The number of th ese
batches and the total number of the related data records are c ounted.
5.4 The Concept Explanation View
A correlation matrix (Figure 1(e)) is employed to support DR4 because of its representation ability [44,48]. The data source is considered as an attribute to label the context of the data records. For other
attributes, the correlation for each batch of data records is quantiﬁed
by the cosine similarity. The attributes are sorted by the average correlation of the selected batches. ConceptExplorer draws correlation
matrices for the data source and analyst-speciﬁed number of the attributes with the highest correlations subject to the conce pt.
For each cell, the horizontal and vertical axes of each matri x are
deﬁned by two attributes. A square in non-diagonal cells rep resents
a set of data records whose two attributes fall into the value ranges
which are encoded with the position of the square. The differ ences
between the number of records with positive labels and those with
negative labels are counted for each square. The ratio of the difference of two counts over their sum ( i.e., #Positives− #Negatives
#Positives+#Negatives ) is
encoded in color (ranging from red to blue). When the label di stribution in the dataset is nonuniform, analysts can reset t he color
mapping and encode the percentage difference in all data rec ords in
white. In some speciﬁc contexts, certain cells may be empty. To distinguish squares without a record, strokes are added in the s quares
with more than one record. Darker strokes imply that the reco rd
number of the square is larger than 5% of the amount of chosen
data records. The matrix view exhibits a symmetrical layout . Taking advantage of this feature, the current correlation patt ern can be
compared with the other one. Each cell on the diagonal presen ts a
pair of histograms (i.e., a grounded histogram for lower-le ft corner
and an inverted histogram for upper-right corner).
5.5 Interactions
Following the workﬂow mentioned in Section 4.2, ConceptExplorer supports the following interactions.
Navigate by overview. Analysts ﬁrst brush a time segment in the
timeline navigator view and check related details in the pre diction
model view and the concept-time view.
Inspect concept drifts. In the accuracy ﬂuctuation chart, the
probability threshold c for the consistency judgment model can be
deﬁned by a slider. To study why a concept drift is emphasized ,
analysts can check the inferred time segments for a data sour ce.
Specify the context of a concept. Analysts can specify a timestamp by dragging the handle in the accuracy ﬂuctuation chart . According to the time stamp, the concept-time view shows the re commended time segments and selects the batches in the time segm ents.
If analysts are unsatisﬁed with the automatically selected batches,
they can adjust the selection ranges by dragging the boundaries. Analysts can choose the data records which need to be included i n the
concept explanation view. Data sources are labeled with “in consistent” and “consistent”. The set of all consistent data so urces is
recommended.
Identify concepts. The data records with the speciﬁed context
are integrated into the correlation matrix. Analysts are al lowed to
set the number of listed attributes. If analysts are interes ted in the
pattern shown in the lower-left corner of the correlation ma trix, i.e.,
the description of the concept with the current selected context, they
can save the screenshot and related concept (see the right of Figure 1(e)) by clicking the “Identify” button.
Compare concepts. In subsequent explorations, analysts can
change the data in the upper-right corner of the correlation matrix.
ConceptExplorer highlights a pair of squares at symmetrical positions for comparison when one of them is speciﬁed by analysts .
6 C ASE STUDIES
We present three case studies based on real-world datasets. V arious
concepts and concept drifts are analyzed to evaluate the eff ectiveness of ConceptExplorer.
6.1 Beijing Air Quality Forecast
In this case, we attempt to understand the dominant factors a ffecting air quality. We employ the air pollutant data [55] from fo ur
nationally-controlled air-quality monitoring sites in Be ijing, which
was collected every hour from March 1st, 2013 to February 28t h,
2017 (34, 536 data records per site). 22 meteorology-related dimensions are applied to predict if the air quality index (AQI) is higher
than 100 ( i.e., worse than mild pollution) after 24 hours.
The timeline navigator view indicates that concept drifts o ccurred almost every few days (Figure 1(b), DR1). The drift levels of all data sources are abnormally stable ( < 2, i.e., the warning
level) during the week at the end of March 2015, except for dat a
source (DS1, i.e., the site at Guanyuan park). We choose a 40-day
time segment around the week. As shown in Figure 1(c), all dat a
sources have a similar ﬂuctuation of the prediction accurac y (DR2).
Each one experienced more than one concept drift between Mar ch
17th and March 20th. We select two time segments before and af -
ter the drift time segment (see the black and blue time segmen ts in
Figure 6, DR3).
As shown in Figure 1(e), the comparison result of two time
segments indicates that their associated concepts have sim ilarities
(DR5). For instance, the higher the PM10 concentration is, the
more records are labeled with poor air quality ( DR4). The difference mainly lies in that more high air quality records are obs erved
(i.e., more blue squares in the upper-right corner) after the drift time
segment. The records with a low Dew Point (i.e., dew point temperature ( ◦C)) are more likely to be labeled with good air quality.
Also, the order of the attributes indicates that the dominan t pollutant PM10 is replaced by PM2.5
day (i.e., the average PM2.5
concentration (µg/m3) in the past 24 hours) after the concept drift.
The concept drift of DS1 occurred on March 26th is identiﬁed
as inconsistent with others by the consistency judgment mod el. Besides, the projected parameter trajectory of DS1 (see Figur e 7) indicates that the parameters of DS1 go through a twist that is dif ferent
from others ( DR2). To explore the inconsistent behavior of DS1,
the time segments after (the red time segments) the concept d rift
on March 26th for DS1 is checked (Figure 6, DR3) . All cells
in the correlation matrix turn into red (see the upper-right corner
highlighted by the red dashed line in Figure 8), which implie s that

## Page 7 / 12

Before the concept drift
After the concept drift
During the sandstorm
DS1: Guanyuan
0
5
10 
15 
20 
Arriving time
4 batches
96 records
Batch size Risk level: 0 5
7 batches
168 records
Mar 02 Apr 10 Mar 22
4 batches
4 batches
96 records
96 records
DS3: Wanshouxigong
0
5
10 
15 
20 
Arriving time
Batch size Risk level: 0 5
4 batches
96 records
10 batches
240 records
Mar 02 Apr 10 Mar 22 
DS2: Tiantan 
0
5
10 
15 
20 
Arriving time
Batch size Risk level: 0 5
4 batches
96 records
10 batches
240 records
Mar 02 Apr 10 Mar 22 
DS4: Dongsi
0
5
10 
15 
20 
Arriving time
Batch size Risk level: 0 5
4 batches
96 records
10 batches
240 records
Mar 02 Apr 10 Mar 22 
The inconsistent data source Consistent data sources
Before the sandstorm
Time segments Time segments 
Figure 6: Details of three time segments analyzed in the ﬁrst case.
almost all records are labeled with poor air quality ( DR4, DR5 ).
The weather records indicate that there was a sandstorm in Be ijing
at the end of March. The dominant pollutant during the time se gment before the sandstorm (see the blue time segments in Figu re 6)
is PM10, as shown in the lower-left corner highlighted by the blue
dashed line of Figure 8, which contributes to the inconsiste nt concept drift.
DS3 DS2DS1
Time: March 22nd, 2014 March 28th, 2014 
DS4
March 24th - March 25th
Figure 7: The parameter projection view between March 22nd, 2014
to March 28th, 2014.
Actually, other data sources record the same sandstorm. The
dominance of PM2.5 has not been replaced by PM10 before the
sandstorm, and thus the concept drift was not triggered by th e sandstorm. Instead, the same label with poor air quality make the prediction task simple for the prediction model. The prediction ac curacy
has risen to 100% in a couple of days. At the beginning of April , the
sandstorm ended, and the air quality detected by all data sources improves, which leads to the next concept drift. An expert work ing in
the meteorological bureau told us that the spring sandstorm s in Beijing are basically caused by PM2.5. Such pollutants are spre ad by
wind. Therefore, the detection results of sites in differen t locations
have slight differences.
6.2 Consumption Behaviors of MMORPG Players
In the second case, we study the dynamics of consumption beha viors in multiplayer online role-playing game (MMORPG) to un derstand game company’s operating strategies. For example, re leasing
a new role may attract new players to join in the game and consume,
which leads to changes in the concept of consumption behavio rs.
: -100% 100% #Data records: 0 <9 >=9 #Positives − #Negatives
#Positives + #Negatives
Figure 8: The correlation matrix compares the two concepts f rom
two time segments of DS1. The lower-left corner (the dashed r egion
in blue) corresponds to the blue time segment in Figure 6. The upper-right corner (the dashed region in red) exhibits the san dstorm
pattern.
The employed dataset contains player records from three ser vers
(647, 800 player records from Server17, 702 , 125 player records
from Server164, and 585 , 048 player records from Server230) of
a MMORPG from August 16th, 2013 to January 19th, 2014. Three
servers were started at different timestamps: Server17, Se ver164,
and Server230, which are in order of time, that is, players on different servers register for the game at different time periods. For each
player, 21 attributes, like equipment (i.e., the combat effectiveness
score of the player’s equipment), practice (i.e., the level of practice,
improved by learning and improving skills and ﬁnishing task s), are
recorded every day. The consumption records for the upcomin g
week of players form a group of time-series.
We ﬁrst browse the entire time span to learn about the evoluti on
of consumption behaviors in three servers. With the thresho ld of
70%, all concept drifts are identiﬁed as inconsistent by the consistency judgment model. Besides, the projection of parameter s of
Server230 is far from those of the parameters of other data so urces
(see Figure 9, DR2), which implies that the consumption behaviors of the players in Sever230 is quite different from the ot her
two servers. In particular, Server164 has a similar traject ory with
Server17 from August 2013 to November 2013. After that, the trace
of Server230 shows a sharp downward turn. The speciﬁc time po int
is further studied. As shown in Figure 10(a), the number of pl ayers
at the moment was doubled—on October 24th, the game operator s
merged Server230 with Server229 to maintain player engagem ent.
We notice that Server230 has fewer concept drifts than the ot her
two servers before this merge, as shown in Figure 10(b). We co me
up with a hypothesis that the consumption behaviors of playe rs in
Server230 affected less by various events than other server s. This
phenomenon may be one reason for the operators to merge serve rs.
To verify this hypothesis, an activity held a month earlier t han
the server merge (see Figure 10(b)) is analyzed. The consist ency
judgment model regards Server230 as inconsistent with the o ther
two servers. We use records from Server17 and Server164 to st udy
players’ respond to this event. The time segments before and during the event are selected separately ( DR3). As shown on the left
of Figure 11, the right-bottom square changes from gray to bl ue,

## Page 8 / 12

Server230Server164Server17Time: 
Figure 9: The overview of the parameter projection view. The orange
circle denotes the moment when Server230 is merged.
0
1,000
2,000
3,000
4,000
5,000
Arriving time
Batch size Risk level: 0 5
Server merge 
(a) The data details view of Server230 
(b) The overview of the timeline navigator view
Before the server merging
After the server merging
The update event
Aug 19 Jan 18Nov 03
Figure 10: Visualizations of a server merging event of Sever 230: (a)
the data details view and (b) the timeline navigator view.
which implies that a certain number of players in Server164 ( DS2)
took the opportunity to update their equipment to the highest level
(DR4, DR5 ). Besides, some high- practice but poorly equipped
players in both servers were enthusiastic about the event an d consumed virtual currency during the event (see the left-top sq uares in
the two cells on the right of Figure 11). However, no signiﬁca nt
changes are observed in Server230.
We invited a data analysis expert, who was in charge of operating
the game, to check our ﬁndings. She told us that because of pla yer
loyalty, the older the server is, the more the enthusiasm for game
events. For newly opened servers, payment peaks occurred ma inly
at the moment of launching. As for merging servers, she told u s that
some players created smurfs in servers to collect equipment or provide assistance after merging servers. And the most efﬁcien t way
to create a high-quality smurf is to consume during events. T hese
observations verify the hypothesis.
6.3 Movie Rating Prediction
To comprehensively learn about the evolution of audience pr eference on different movies, we study whether the average ratin g of
a movie will increase in the next seven days from three platfo rms:
Rotten Tomatoes [4] (recorded reviews from critics), IMDB ( collected from Twitter) [1], and MovieLens [2]. By extracting d ata
stamped in the common time segment (from February 28th, 2013 to
Practice
0
100K
200K
300K
400K
500K
600K
700K
Equipment 
1
30 
60 
90 
120 
150 
1
30 
60 
90 
120 
150 
Data SourceDS1 
DS2 
DS1 
DS2 
Before the update event After the update event
#Data records: 0 <3
616 >=3,616
: -100% 100%#Positives − #Negatives 
#Positives + #Negatives 
Figure 11: Patterns before and after the update event. Blue i ndicates
that more players have consumption in the upcoming seven day s.
March 31st, 2015), there are 96174, 385015, 1127948 records from
three sources, respectively. Each record includes rating d ate, rating score, and movie ID. The movie ID is replaced with the movi e
description [3], like the release year, budget, duration, etc. The
training data for the prediction model has 15 dimensions.
Our analysis starts from the summer vacation because most pe ople have chances to watch movies during this period. We selec t the
segment from June 15th, 2014 to July 10th, 2014 from the timeline navigator view. As shown in Figure 12, the prediction mo dels
trained by the data records from different data sources have distinct
accuracy ﬂuctuations (DR2). The consistency judgment model suggests to study three data sources separately. After groupin g tests
(DR3), we ﬁnd that the records from Rotten Tomatoes and IMDB
(the lower-left corner highlighted by the yellow dashed lin e) show
clearer patterns than those from MovieLens (the upper-righ t corner
highlighted by the brown dashed line, DR5), as shown in Figure 13.
By studying red squares in the lower-left corner, we detect t hree
descriptions corresponding to the movies whose ratings hav e declined: movies whose release year is 2014, movies with relatively
high budgets and action movies (DR4). Moreover, squares corresponding to each intersection of the above descriptions are in conspicuous red. The reason may be that a highly anticipated mov ie
does not meet audience expectations. The rating records tur n out
that the disappointing movie is Transformers: Age of Extinction .
We further observing the accuracy curves to learn platform c haracteristics. It can be seen from Figure 12 that the accuracy ﬂ uctuation of Rotten Tomatoes is more severe than others. Espec ially,
there exists periodic ﬂuctuations in the curve ( DR2). Concept drifts
appeared once in about a week. The number of arriving data records
has the same periodicity, as shown in Figure 14(a). We select the
time segments of the previous week and the next week ( DR3). The
main difference is identiﬁed from movies released in the 199 0s and
2000s (see Figure 14(b), DR4, DR5): new ratings reduce the average ratings of certain old movies. Similar patterns are no t found
from other data sources. This may be caused by speciﬁc recommendations from the Rotten Tomatoes—we notice that there ar e
sections for “hidden gem movies” on the Rotten Tomatoes webs ite.
7 D ISCUSSION
In this section, we summarize the feedback from three expert s and
discuss the considerations of our approach.
7.1 Expert Reviews
We invited three professors in related ﬁelds as experts to re view our
system. The ﬁrst expert (E1) has been working on massive data

## Page 9 / 12

MovieLens Rotten Tomatoes IMDB
 Warnings ğ Inconsistent Consistent Confirmed drifts ğ
Severe accuracy drops of Rotten Tomatoes 
Figure 12: The accuracy ﬂuctuation of the prediction models trained by the data from three data sources (June 15th, 2014 - July 10th, 2014).
: -100% 100% 
 #Data records: 0 <458 >=458 #Positives − #Negatives 
#Positives + #Negatives 
Figure 13: The correlation matrix displays concepts extrac ted from
the segment around June 28th, 2014. The lower-left corner (t he
dashed region in yellow) shows records from Rotten T omatoes (DS2)
and IMDB (DS3). The upper-right corner (the dashed region in
brown) shows those from MovieLens (DS1). Due to the uneven di stribution of labels, the color mapping of the correlation ma trix is reset
to map the average difference to white.
#Data records: 0 <40 >=40
#Difference: -100% 100%
Fri 13 Thu 03 Wed 23 
0
50 
100
150
200
250
Arriving time
7 batches
602 records
6 batches
525 records
Batch size Risk level: 0 5
DS2: Rotten Tomatoes 
1900
1940
1980
1990
2000
2005
2010
2013
2014
2015
Year 
/ The previous week / The next week
(a) The data details view (b) Partial correlation matrix
Figure 14: The details of data records collected from Rotten T omatoes in two weeks. (a) The data details view shows two speciﬁc time
segments. (b) T wo cells of the correlation matrix depict the correlation between year and the concept.
analysis for twelve years. The other two experts (E2 & E3) hav e
at least seven years of experience in visual analytics of tim e-series
data. A semi-structured interview was conducted with each e xpert
through a remote conference. We ﬁrst introduced our method a nd
visual design in about 20 minutes. Then, we showed them case
studies, during which they were free to ask questions and exp ress
opinions. We summarized their feedbacks as follows.
Effectiveness. All experts agree with the effectiveness of our
approach. “The concept drift index can indeed reﬂect the cha nge
process of the transformation data distribution over time to a certain
extent,” E1 commented. E2 and E3 also appreciated our idea of
applying the drift level index. E3 said that the index can eff ectively
support interactive exploration of unknown concepts and co ncept
drifts.
Scalability. E1’s main concern is whether high-dimensional
data, i.e., data with hundreds of dimensions, can be applied in our
system. Through system demo, we proved to him that our visual
analysis approach is minimally affected by the curse of dime nsion.
Related discussion can be found in Section 7.3. In summary, h e
believes that our workﬂow and system can meet the need for ana -
lyzing time-series data and he would like to use our system when he
has related analysis requirements. For further extension o f our approach, he gave us two suggestions: 1) considering a multi-m odel
hybrid prediction method to enhance the reliability of the c oncept
drift index; 2) recommending concepts or concept drifts bas ed on
data features automatically.
Visual Designs. Concerning the interface design, all experts
gave positive feedbacks. E2 particularly likes the hierarc hical abstraction in the timeline navigator view and the prediction model
view. E3 was impressed by the shifted stripes in the predicti on
model view.
Learn costs. E2 and E3 commented that analysts need time to
learn before they can use the system. Considering that the deﬁnition
and visual representation of concept drift are abstract and complex,
they agree that it does worth learning costs.
Advice. Considering that the analysis may only involve partial
data sources, E2 suggested supporting the ﬁlter of data sour ces in
the data entrance, which can facilitate analysts to focus on certain
data sources. We update the system and allow analysts to cont rol
the display of data sources. However, the training of the consistency
judgment model can not be completed interactively. Analyst s have
to reset data sources from the backend to modify the model res ults.
7.2 The Navigation of the Drift Level Index
The drift level index provides analysts with comprehensive navigation of various dynamics of concepts by connecting predic tion
models and visual analysis of time-series data. However, no t all
dynamics are identiﬁed by the drift level index. As mentione d in
Section 3.3, the computation of the drift level index ignore s the situations that the accuracy of predictive models is increasing or stable.
The understanding of concepts keeps updating with iteratio ns, even
when the accuracy does not drop. The dynamics that can not be

## Page 10 / 12

reﬂected from the drift level index mainly fall into two cate gories:
improvements during learning processes and slow changes th at can
be caught by iterations.
Prediction models initialize their understanding of conce pts (i.e.,
parameters) at the beginning of the training process. The su bsequent iterations always contribute to a rapid rise of accura cy. The
same phenomena appear when the adaptive mechanics (replaci ng
the weakest prediction model) are triggered to stop the accu racy declines caused by concept drifts. In other words, the results of these
changes can be observed by inspecting the concepts followin g related concept drifts. In addition, concepts may evolve slow ly. If
prediction models can follow the changes by accumulative up dates
in iterations, no concept drift can be detected. To reveal th e imperceptible changes, parameter evolution is monitored by the p arameter projection view, which not only provides an overview of the
entire learning process but also indicates the accumulativ e changes.
7.3 Scalability
7.3.1 Visual Designs
We discuss the visual scalability issue from the following a spects.
Data records. ConceptExplorer assists analysts to locate appropriate contexts of concepts step by step, during which no att ention
needs to be paid on single data records or their attribute val ues. Because the dynamic features of data records distributed in di fferent
contexts are extracted by automatic approaches. The concep ts corresponding to the selected contexts are summarized by the di fferences in the number of records with positive labels and negat ive
ones. Analysts can contribute to qualitative conclusions b ased on
the distribution of differences over an attribute or a pair of attributes,
as mentioned in Section 6.
Attributes. Due to the limitation of display space, up to 15
attributes are shown in the concept explanation view. To pro vide
signiﬁcant patterns with sufﬁcient spaces, only attribute s with top
correlations with the label are listed.
Data sources. Shifted stripes are employed to eliminate visual
clutter caused by multiple data sources in the accuracy ﬂuct uation
chart. The gap width of stripes can be increased to insert mor e
lines, i.e., adapt to more data sources. In addition, the color map
that encodes data sources should also be adapted to the incre asing
number of data sources.
7.3.2 Computation Time
The performance of models are tested on a desktop with 16G mem -
ory and two Intel Core i7 6700 at 3.4 GHz and 3.41GHz processors (see Table 1). Data records from four data sources used i n the
ﬁrst case are composed into an 88-dimensional data source, n amed
Case1mixed . It can be seen that the size of data affects the performance of prediction model training. The computation time of drift
level indices is not affected by the data dimension, but is re lated
to the size of sliding windows. In this work, the size of slidi ng
windows is determined according to the update frequency of d ata
records.
In summary, the time-consuming part of automatic approache s
is training prediction models. In the current version of ConceptExplorer, training and verifying are completed in the preprocessing
stage. With the help of powerful computing clusters or cloud computing, it is possible to extend our system to process massiv e in
real-time data. Hence, our system design and workﬂow have ad equate scalability in terms of data records and attributes.
8 C ONCLUSION
In this paper, we propose a visual analysis approach to facil itate
the exploration of concept drifts from multi-source time-s eries data.
Analysts are allowed to ﬂexibly identify and compare the con cepts
with different contexts. The gradually progressive speciﬁ cation of
the contexts is navigated by the model-derived drift level i ndex and
T able 1: Average computation time (in milliseconds) of an iteration for
the three stages with different combinations of attribute a mount. The
size of sliding window of drift level index is labeled in brac kets.
Data source name
(#Attribute)
Prediction
model
Drift level
index calculation
(size)
Consistency
judgment
model
MovieLens (15) 2.532 0.088 (1500) 0.028
RottenTomatoes (15) 2.698 0.013 (100) 0.018
IMDB (15) 2.542 0.036 (500) 0.015
Guanyun (22) 3.152 0.009 (100) 0.034
Tiantan (22) 3.203 0.009 (100) 0.022
Case1mixed (88) 7.681 0.009 (100) 0.032
the consistency judgment model, which correspond to time se gments and the set of data sources, respectively. A visual ana lysis
system, ConceptExplorer, is designed and implemented.
The effectiveness of ConceptExplorer is veriﬁed through three
case studies with various real-world data sets. In addition , positive
reviews are received from two experts on related ﬁelds. In th e future, we plan to improve the concept explanation view to expl ain
the relationship between attributes and the label in a more c omprehensive way.
ACKNOWLEDGMENTS
This work was supported by National Natural Science Foundation of China (61772456, 61761136020, 61972122, 61872389) and
Open Project Program of State Key Lab of CAD&CG (A1903).
This work has partially been supported by the FFG, Contract N o.
854184: “Pro 2Future is funded within the Austrian COMET Program Competence Centers for Excellent Technologies under t he
auspices of the Austrian Federal Ministry of Transport, Inn ovation
and Technology, the Austrian Federal Ministry for Digital a nd Economic Affairs and of the Provinces of Upper Austria and Styri a.
COMET is managed by the Austrian Research Promotion Agency
FFG.”
REFERENCES
[1] A live movie rating dataset collected from twitter.
https://github.com/sidooms/MovieTweetings.
[2] MovieLens 20M dataset. https://www.kaggle.com/grouplens/movielens-20m-data set.
[3] The movies dataset. https://www.kaggle.com/rounakbanik/the-movies-datas et.
[4] Rotten Tomatoes datasets. https://www.kaggle.com/stefanoleone992/rotten-tomat oes-movies-and-critics-datasets .
[5] B. Bach, C. Shi, N. Heulot, T. Madhyastha, T. Grabowski, a nd P . Dragicevic. Time curves: Folding time to visualize patterns of t emporal
evolution in data. IEEE Transactions on Visualization and Computer
Graphics, 22(1):559–568, 2015.
[6] G. Baltussen, S. van Bekkum, and Z. Da. Indexing and stock market
serial dependence around the world. Journal of Financial Economics,
132(1):26–48, 2019.
[7] H. Becker and M. Arias. Real-time ranking with concept dr ift using
expert advice. In Proceedings of the 13th ACM SIGKDD , pp. 86–94,
2007.
[8] D. Brzezinski and J. Stefanowski. Combining block-base d and online
methods in learning ensembles from concept drifting data st reams. Information Sciences, 265:50–67, 2014.
[9] A. P . Cassidy and F. A. Deviney. Calculating feature impo rtance in
data streams with concept drift using online random forest. In 2014
IEEE International Conference on Big Data , pp. 23–28, 2014.
[10] D. Ceneda, T. Gschwandtner, T. May, S. Miksch, H.-J. Sch ulz,
M. Streit, and C. Tominski. Characterizing guidance in visu al analytics. IEEE Transactions on Visualization and Computer Graphics ,
23(1):111–120, 2016.
[11] D. R. de Lima Cabral and R. S. M. de Barros. Concept drift d etection based on Fishers Exact test. Information Sciences, 442:220–234,
2018.

## Page 11 / 12

[12] J. Demˇ sar and Z. Bosni´ c. Detecting concept drift in data streams using
model explanation. Expert Systems with Applications , 92:546–559,
2018.
[13] J. Demˇ sar, Z. Bosni´ c, and I. Kononenko. Visualizatio n and concept
drift detection using explanations of incremental models. Informatica,
38(4):321–327, 2014.
[14] G. Ditzler, M. Roveri, C. Alippi, and R. Polikar. Learni ng in nonstationary environments: A survey. IEEE Computational Intelligence
Magazine, 10(4):12–25, 2015.
[15] D. M. dos Reis, P . Flach, S. Matwin, and G. Batista. Fast u nsupervised
online drift detection using Incremental Kolmogorov-Smirnov test. In
Proceedings of the 22nd ACM SIGKDD , pp. 1545–1554, 2016.
[16] A. Endert, W. Ribarsky, C. Turkay, B. W. Wong, I. Nabney, I. D.
Blanco, and F. Rossi. The state of the art in integrating mach ine learning into visual analytics. Computer Graphics F orum, 36(8):458–486,
2017.
[17] J. Gama, P . Medas, G. Castillo, and P . Rodrigues. Learni ng with drift
detection. In Brazilian Symposium on Artiﬁcial Intelligence , pp. 286–
295. Springer, 2004.
[18] J. Gama, I. ˇZliobait˙ e, A. Bifet, M. Pechenizkiy, and A. Bouchachia. A
survey on concept drift adaptation. ACM Computing Surveys, 46(4):1–
37, 2014.
[19] A. Hinterreiter, C. Steinparz, M. Sch¨ oﬂ, H. Stitz, and M. Streit. Exploring visual patterns in projected human and machine deci sionmaking paths. arXiv preprint arXiv:2001.08372, 2020.
[20] N. Hochman and R. Schwartz. Visualizing instagram: Tra cing cultural visual rhythms. In Proceedings of the Sixth International AAAI
Conference on W eblogs and Social Media, 2012.
[21] D. J¨ ackle, F. Fischer, T. Schreck, and D. A. Keim. Tempo ral MDS
plots for analysis of multivariate data. IEEE Transactions on Visualization and Computer Graphics , 22(1):141–150, 2015.
[22] E. Keogh, S. Chu, D. Hart, and M. Pazzani. Segmenting tim e series:
A survey and novel approach. In Data mining in time series databases,
pp. 1–21. World Scientiﬁc, 2004.
[23] P .-M. Law, W. Wu, Y . Zheng, and H. Qu. VisMatchmaker: Coo peration of the user and the computer in centralized matching adjustment. IEEE Transactions on Visualization and Computer Graphics ,
23(1):231–240, 2016.
[24] T.-Y . Lee and H.-W. Shen. Visualization and exploratio n of temporal trend relationships in multivariate time-varying data . IEEE Transactions on Visualization and Computer Graphics , 15(6):1359–1366,
2009.
[25] B. Li, P . Zhao, S. C. Hoi, and V . Gopalkrishnan. PAMR: Pas sive
aggressive mean reversion strategy for portfolio selectio n. Machine
learning, 87(2):221–258, 2012.
[26] J. Li, S. Chen, K. Zhang, G. Andrienko, and N. Andrienko. COPE:
Interactive exploration of co-occurrence patterns in spat ial time series. IEEE Transactions on Visualization and Computer Graphics ,
25(8):2554–2567, 2018.
[27] Y . Liang, X. Wang, S.-H. Zhang, S.-M. Hu, and S. Liu. Phot oRecomposer: Interactive photo recomposition by cropping. IEEE Transactions on Visualization and Computer Graphics , 24(10):2728–2742,
2017.
[28] A. Liu, J. Lu, F. Liu, and G. Zhang. Accumulating regiona l density
dissimilarity for concept drift detection in data streams. Pattern Recognition, 76:256–272, 2018.
[29] D. Liu, P . Xu, and L. Ren. TPFlow: Progressive partition and multidimensional pattern extraction for large-scale spatio- temporal data
analysis. IEEE Transactions on Visualization and Computer Graphics, 25(1):1–11, 2018.
[30] M. Liu, J. Shi, K. Cao, J. Zhu, and S. Liu. Analyzing the tr aining processes of deep generative models. IEEE Transactions on Visualization
and Computer Graphics, 24(1):77–87, 2018.
[31] S. Liu, W. Cui, Y . Wu, and M. Liu. A survey on information v isualization: recent advances and challenges. The Visual Computer ,
30(12):1373–1393, 2014.
[32] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang. Learni ng under
concept drift: A review. IEEE Transactions on Knowledge and Data
Engineering, 31(12):2346–2363, 2018.
[33] Y . Lu, R. Garcia, B. Hansen, M. Gleicher, and R. Maciejew ski. The
state-of-the-art in predictive visual analytics. Computer Graphics F orum, 36(3):539–562, 2017.
[34] C. Niederer, H. Stitz, R. Hourieh, F. Grassinger, W. Aig ner, and
M. Streit. TACO: visualizing changes in tables over time. IEEE
Transactions on Visualization and Computer Graphics , 24(1):677–
686, 2017.
[35] K. Nishida and K. Y amauchi. Detecting concept drift usi ng statistical
testing. In Proceedings of the International Conference on Discovery
Science, pp. 264–269. Springer, 2007.
[36] I. Olier and A. V ellido. Capturing the dynamics of multi variate time
series through visualization using generative topographi c mapping
through time. In 2006 IEEE International Conference on Engineering of Intelligent Systems , pp. 1–6.
[37] K. B. Pratt and G. Tschapek. Visualizing concept drift. In Proceedings
of the Ninth ACM SIGKDD , pp. 735–740, 2003.
[38] J. C. Schlimmer and R. H. Granger. Incremental learning from noisy
data. Machine learning, 1(3):317–354, 1986.
[39] T. S. Sethi, M. Kantardzic, and H. Hu. A grid density base d framework
for classifying streaming data in the presence of concept dr ift. Journal
of Intelligent Information Systems , 46(1):179–211, 2016.
[40] G. Shurkhovetskyy, N. Andrienko, G. Andrienko, and G. F uchs. Data
abstraction for visualizing large time series. In Computer Graphics
F orum, vol. 37, pp. 125–144. Wiley Online Library, 2018.
[41] C. A. Steed, W. Halsey, R. Dehoff, S. L. Y oder, V . Paquit, and S. Powers. Falcon: Visual analysis of large, irregularly sampled , and multivariate time series data in additive manufacturing. Computers &
Graphics, 63:50–64, 2017.
[42] G. Stiglic and P . Kokol. Interpretability of sudden con cept drift in
medical informatics domain. In Proceedings of the 2011 IEEE 11th
International Conference on Data Mining W orkshops, pp. 609–613.
[43] H. Stitz, S. Gratzl, W. Aigner, and M. Streit. ThermalPl ot: Visualizing
multi-attribute time-series data using a thermal metaphor. IEEE Transactions on Visualization and Computer Graphics , 22(12):2594–2607,
2015.
[44] J. Suschnigg, B. Mutlu, A. K. Fuchs, V . Sabol, S. Thalman n, and
T. Schreck. Exploration of anomalies in cyclic multivariat e industrial
time series data for condition monitoring. In Proceedings of the 3rd
International W orkshop on Big Data Visual Exploration and Analytics,
2020.
[45] G. Tkachev, S. Frey, and T. Ertl. Local prediction model s for spatiotemporal volume visualization. IEEE Transactions on Visualization
and Computer Graphics, 2019.
[46] J. Walker, R. Borgo, and M. W. Jones. TimeNotes: A study o n effective chart visualization and interaction techniques fo r time-series
data. IEEE Transactions on Visualization and Computer Graphics ,
22(1):549–558, 2015.
[47] M. E. Wall, A. Rechtsteiner, and L. M. Rocha. Singular va lue decomposition and principal component analysis. In A practical approach to
microarray data analysis, pp. 91–109. Springer, 2003.
[48] X. Wang, J.-K. Chou, W. Chen, H. Guan, W. Chen, T. Lao, and
K.-L. Ma. A utility-aware visual approach for anonymizing m ultiattribute tabular data. IEEE Transactions on Visualization and Computer Graphics, 24(1):351–360, 2017.
[49] G. I. Webb, L. K. Lee, B. Goethals, and F. Petitjean. Anal yzing concept drift and shift from sample data. Data Mining and Knowledge
Discovery, 32(5):1179–1199, 2018.
[50] G. Widmer and M. Kubat. Learning in the presence of conce pt drift
and hidden contexts. Machine Learning, 23(1):69–101, 1996.
[51] Y . Wu, Z. Chen, G. Sun, X. Xie, N. Cao, S. Liu, and W. Cui. St reamExplorer: A multi-stage system for visually exploring even ts in social
streams. IEEE Transactions on Visualization and Computer Graphics,
24(10):2758–2772, 2018.
[52] W. Y ang, Z. Li, M. Liu, Y . Lu, K. Cao, R. Maciejewski, and S . Liu.
Diagnosing concept drift with visual analytics. In IEEE Conference
on Visual Analytics Science and Technology, 2020.
[53] Y . Y ao, L. Feng, and F. Chen. Concept drift visualizatio n. Journal of
Information &Computational Science, 10(10):3021–3029, 2013.
[54] J. Y uan, C. Chen, W. Y ang, M. Liu, J. Xia, and S. Liu. A surv ey
of visual analytics techniques for machine learning. Computational
Visual Media, 7(1):1–31, 2021.

## Page 12 / 12

[55] S. Zhang, B. Guo, A. Dong, J. He, Z. Xu, and S. X. Chen. Caut ionary tales on air-quality improvement in Beijing. Proceedings of the
Royal Society A: Mathematical, Physical and Engineering Sc iences,
473(2205):20170457, 2017.
