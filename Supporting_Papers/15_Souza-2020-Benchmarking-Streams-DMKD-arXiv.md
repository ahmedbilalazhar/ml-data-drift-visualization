# Benchmarking Stream Learning: On the (In)ability to Evaluate Streaming Algorithms

**Authors:** Vinicius M. A. Souza, Denis M. dos Reis, Andre G. Maletzke, Gustavo E. A. P. A. Batista

**Venue:** Data Mining and Knowledge Discovery (2020), arXiv version — Benchmark methodology reference

*Source PDF: `15_Souza-2020-Benchmarking-Streams-DMKD-arXiv.pdf`*

*Converted to markdown following the Selected_Papers strategy (full-text extraction, page-ordered). Verify title/authors/year against publisher before citing.*

---

## Page 1 / 54

Data Mining and Knowledge Discovery manuscript No.
(will be inserted by the editor)
Challenges in Benchmarking Stream Learning
Algorithms with Real-world Data
Vinicius M. A. Souza · Denis M. dos Reis ·
Andre G. Maletzke · Gustavo E. A. P. A.
Batista
Received: date / Accepted: date
Abstract Streaming data are increasingly present in real-world applications such
as sensor measurements, satellite data feed, stock market, and ﬁnancial data. The
main characteristics of these applications are the online arrival of data observations at high speed and the susceptibility to changes in the data distributions due
to the dynamic nature of real environments. The data stream mining community
still faces some primary challenges and diﬃculties related to the comparison and
evaluation of new proposals, mainly due to the lack of publicly available nonstationary real-world datasets. The comparison of stream algorithms proposed in
the literature is not an easy task, as authors do not always follow the same recommendations, experimental evaluation procedures, datasets, and assumptions.
In this paper, we mitigate problems related to the choice of datasets in the experimental evaluation of stream classiﬁers and drift detectors. To that end, we
propose a new public data repository for benchmarking stream algorithms with
real-world data. This repository contains the most popular datasets from literature and new datasets related to a highly relevant public health problem that
involves the recognition of disease vector insects using optical sensors. The main
advantage of these new datasets is the prior knowledge of their characteristics and
patterns of changes to evaluate new adaptive algorithm proposals adequately. We
also present an in-depth discussion about the characteristics, reasons, and issues
V. M. A. Souza
Universidade de S˜ ao Paulo, S˜ ao Carlos, SP, Brazil
University of New Mexico, Albuquerque, NM, USA
E-mail: vinicius@unm.edu
D. M. dos Reis
Universidade de S˜ ao Paulo, S˜ ao Carlos, SP, Brazil
E-mail: denismr@usp.br
A. G. Maletzke
Universidade de S˜ ao Paulo, S˜ ao Carlos, SP, Brazil
E-mail: andregustavo@usp.br
G. E. A. P. A. Batista
University of New South Wales, Sydney, Australia
Universidade de S˜ ao Paulo, S˜ ao Carlos, SP, Brazil
E-mail: g.batista@unsw.edu.au
arXiv:2005.00113v2 [cs.LG] 30 Jun 2020

## Page 2 / 54

2 V. M. A. Souza et al.
that lead to diﬀerent types of changes in data distribution, as well as a critical
review of common problems concerning the current benchmark datasets available
in the literature.
Keywords Data stream · Concept drift · Classiﬁcation· Drift detection ·
Benchmark data
1 Introduction
In the last 20 years, we have witnessed the emergence and notable increase in
the interest of algorithms that learn from streaming data. This new generation
of machine learning methods is designed to deal with continuous ﬂows of data.
Frequently, such streams comprise changes in the distribution of data, which are
governed by the dynamics of real-world problems and application domains that
evolve. In the context of machine learning, these changes in data distribution are
named concept drifts (Widmer and Kubat, 1996) and typically occur in data that
are observed continuously at a fast rate, which in turn impose time and memory
constraints on the algorithms that process them.
Batch learning is the standard machine learning approach that assumes the
whole dataset is available at training time. Batch learning is a mature ﬁeld with
clear procedures to evaluate and compare diﬀerent methods using a vastitude of
data shared by researchers for benchmarking. However, in the online scenario of
data stream mining, we still face some primary challenges and diﬃculties related
to the comparison and evaluation of new proposals, mainly due to the lack of
publicly available non-stationary real-world datasets. For example, we found more
than 300 stationary datasets for classiﬁcation problems from diﬀerent domains in
the UCI Machine Learning Repository (Dua and Graﬀ, 2017). In particular, for
time-series classiﬁcation, the UEA & UCR Time Series Classiﬁcation Repository
(Bagnall et al., 2019) stores more than 100 datasets and 20 algorithms from the
literature. For data stream mining, although there is the popular open-source
framework MOA (Bifet et al., 2010a) with a collection of algorithms and synthetic
data generators, we do not have any public repository with a reasonably-sized
collection of real-world stream datasets accompanied of their detailed description.
More alarming, diﬀerent data stream algorithms often run on speciﬁc assumptions
about the data (for instance, methods may assume changes to be either incremental
or abrupt). However, frequently it is not clear whether employed datasets fulﬁll
such assumptions or not.
As recently noted by Krawczyk et al. (2017), the comparison of stream learning
algorithms is not an easy task, as authors do not always follow the same recommendations, experimental evaluation procedures, datasets, and assumptions. In
this paper, we want to mitigate the problems related to the choice of datasets in
the experimental evaluation of stream classiﬁers and drift detectors.
In summary, the main contributions of this article are the following:
– Presentation of basic concepts of data stream mining accompanied by an indepth discussion about the characteristics, reasons, and issues, which lead to
diﬀerent types of changes in data distribution;

## Page 3 / 54

Challenges in Benchmarking Stream Learning 3
– A review of the main real-world datasets adopted in the evaluation of stream
learning methods accompanied by a critical discussion of issues concerning the
data and the challenges imposed due to the lack of a benchmark standard;
– Presentation of a relevant public health problem that involves the recognition
of disease vector insects by an optical sensor, which is responsible for generating
evolving data over time;
– Building (data collection, preprocessing, and features extraction) of 11 new
real-world datasets with controlled concept changes where it is possible to
identify the types/patterns of changes and when they occur for each dataset.
Such data are accompanied by an experimental evaluation that includes stateof-the-art classiﬁers and drift detectors;
– Development and availability of a repository 1 with 27 real-world datasets for
benchmarking the evaluation of stream classiﬁers and change detectors.
This paper is organized as follows. In Section 2, we provide a background
on data stream and concept drift, as well as a discussion about characteristics,
reasons, and issues that lead to changes in the data distribution. In Section 3,
we present an overview of the most common datasets used in the evaluation of
stream mining approaches. In Section 4, we discuss the challenges faced by the
stream mining community when the currently most popular real-world datasets are
evaluated. In Section 5, we introduce a benchmark dataset for stream learning with
controlled concept drifts, which are generated by an optical sensor that measures
characteristics of insect ﬂights. In Section 6, we present to the stream learning
community a new public repository with an initial amount of 27 datasets. In
Section 7, we perform an evaluation followed by a discussion concerning the 11
datasets introduced in this paper. Finally, we conclude our work in Section 8.
2 Background
In this section, we present the main concepts and deﬁnitions regarding data
streams, concept drift and the tasks of classiﬁcation and drift detection under
non-stationary environments.
2.1 Data Stream
A data stream is an ordered sequence of instances continuously observed over time.
Streaming data are increasingly prevalent in real-world applications. Representative examples include network traﬃc, database transactions, sensor measurements,
satellite data feed, stock market and ﬁnancial data, georeferenced data from mobile
devices, among others.
Formally, a data stream is a sequence of instancesDS ={X1, X2, . . . ,Xt, . . .},
where Xt ∈X is a d-dimensional vector in the feature space observed at time
t. In practice, Xt is an ordered list of descriptive attributes that represent the
observation being made. The attributes can be qualitative (nominal, ordinal, or
binary) or quantitative (discrete or continuous).
1 USP Data Stream Repository – Available online at https://sites.google.com/view/
uspdsrepository.

## Page 4 / 54

4 V. M. A. Souza et al.
Among possible tasks such as clustering, regression, graph-mining, outlier detection, and recommender systems, classiﬁcation is probably the most prominent
task on data streams and the focus of this work. In classiﬁcation problems, each
instance Xt∈X is associated to a class label yt∈Y , whereY contains l possible
labels,Y ={C1, C2, . . . , Cl}. Therefore, a classiﬁcation data stream is a sequence
of pairsDS ={(X1, y1), (X2, y2), . . . ,(Xt, yt), . . .}.
Due to a data stream’s potentially inﬁnite length, traditional batch methods
are often not applicable (Gama and Gaber, 2007). These methods typically fail to
comply with at least one of the three most prominent restrictions of the stream
setting (Bifet, 2009):
1. In general, it is impractical to store all events of a stream due to its potentially
inﬁnite length. Only a small portion can be retained in memory;
2. Fast-paced streams require each event to be processed in real-time and, afterwards, discarded; and
3. The underlying distribution of the data may change over time. Hence, old
data can become irrelevant or even detrimental to model the current concept.
Contrary to batch learning, in data stream, we expect the characteristics of
the newly observed data to change when compared to past data.
The ﬁrst constraint limits the amount of memory the algorithms can use, and
the second constraint limits the time that is available for processing each event.
Therefore, the ﬁrst two restrictions lead to the development of techniques that
reduce the information in a stream of data, such as sampling (Chaudhuri et al.,
1999), sketching (Alon et al., 1999), histograms (Gilbert et al., 2002), wavelets
(Matias et al., 2000), and sliding windows (Datar et al., 2002). The third constraint
guides the development of algorithms that are capable of detecting changes in the
data and reacting by updating the existing models.
The non-stationarity of many real environments may lead to changes in the
underlying distribution of the observed data, a phenomenon that goes by many
names in literature, where the most common is concept drift (Widmer and Kubat,
1996). According to the same terminology, the data distribution in a given moment
is called concept. Additionally, a change in its parameters is a drift.
Drifts constitute a central issue since they can decrease the performance of machine learning models induced with historical data (Pan and Yang, 2009; QuioneroCandela et al., 2009; Saenko et al., 2010; Ben-David et al., 2007). A closely related
problem in batch learning is concept shift, which occurs when a model is trained
with data from one distribution and is later applied on data that follow a different distribution (Moreno-Torres et al., 2012). Similarly, transfer learning aims
to extract the knowledge from a source domain where abundant labeled data are
available and applies this knowledge to a related target domain in which insuﬃcient labeled data are available (Pan and Yang, 2009).
2.2 Concept Drift
Concept drifts may manifest with diﬀerent velocity, severity, and patterns. To
illustrate diﬀerent patterns, consider a concept represented by the color and shape
of a geometric ﬁgure. Fig. 1 illustrates three types of drifts discussed in this work:
abrupt, gradual, and incremental.

## Page 5 / 54

Challenges in Benchmarking Stream Learning 5
Abrupt
Gradual
Incremental
Time
Fig. 1 Representation of three types of concept drift over time.
An abrupt drift occurs when the underlying distribution of the data suddenly
changes into a diﬀerent distribution. In other words, after an abrupt transition between two observations, all new data points belong to a concept diﬀerent from the
previous one. In the incremental change, there are several intermediary concepts
between one initial concept and a ﬁnal concept. Consecutive concepts within this
transition period may be indistinguishable. In the case of gradual concept drift, the
transition between two concepts occurs smoothly. However, diﬀerently from incremental drift, in a gradual drift, the probability of observing instances that belong
to the previous concept decreases over time while, simultaneously, the probability
of observing instances that belong to the new concept increases, even though both
concepts are remarkably distinct and stationary during the transition period. Concepts that were seen in the past and are later observed again are called recurring
concepts. We note that one-oﬀ random deviations in the data, such as outliers or
noise, are not considered to be concept drifts.
A practical way to identify such patterns of change is to analyze the data
distribution over sliding windows in the stream. A window represents a sample
of examples that are observed in sequence within a period. When we move the
boundaries that deﬁne the ﬁrst and last data points of this window over the stream
to comprise diﬀerent intervals, we have a sliding window. We note that the use
of windows imposes the choice of essential parameters. The most common is the
number of data points that will be comprised by the window, and how much is
the overlap between consecutive windows.
We can only indirectly observe the underlying concept of the data by analyzing
samples of instances in the sliding window. Therefore, the observation of concept
drift is also indirect. As the number of instances is ﬁnite, there is a discrete and
ﬁnite number of observable distributions that can be analyzed. Note that which
examples are included in the window change the perception of the distribution:
a bigger window can hide inner distributions that would be perceived as distinct
with smaller windows. On the other hand, it may be infeasible to recognize certain
concepts if we can only observe too few data points for each occurring concept.
2.3 Independent Distribution
In batch learning, a common assumption is that examples are independent and
identically distributed (i.i.d.). In data stream applications, such an assumption
usually does not conform to reality. Identically distributed means that the joint
distribution of an example and its class label is the same at any time, that is,

## Page 6 / 54

6 V. M. A. Souza et al.
P (Xt, yt) = P (Xt′ , yt′), when t̸= t′. Meanwhile, independently distributed means
that the probability of the current label does not depend on what was observed
before; that is, P (yt) = P (yt|yt−k).
Since data streams are generated in dynamic environments, examples are not
identically distributed due to the occurrence of concept drifts. Additionally, while
most of the literature assumes independence between examples, some recent studies have found a signiﬁcant temporal dependence in many real-world data (Bifet
et al., 2013; Zliobaite et al., 2015). This dependence on historical class labels has
a direct impact on the design and evaluation of stream approaches.
The existence of temporal dependence reinforces that the line separating data
stream from time-series is blurred. One possible view for streams is that the instances are independent of each other in the sense that the occurrence or absence of
one particular instance implies neither the presence or absence of other particular
instances (Reis et al., 2018a). However, the observation of any instance is under
the inﬂuence of a shared background concept, which is not directly observable.
Even though the instances are not particularly dependent on each other, the occurrence of one instance may be indicative of the likelihood of observing instances
of a particular class or in a speciﬁc region of the feature space, due to this common
background concept. One example is motion recognition. In this problem, sensors
attached to a particular participant may indicate that this person is performing
the eating activity. Since this assessment can be an indication that it is lunchtime,
the likelihood of observing other people performing the same activity may increase,
albeit all the analyzed people being unrelated.
A diﬀerent view, closer to time-series, is that one attribute value is the result
of an auto-regressive transformation applied to previous instances. Examples are
the variation in the price of commodities such as electricity (Zliobaite, 2013) and
the evolution of weather (Ditzler and Polikar, 2013).
Datasets can mix the two views mentioned above by combining features from
diﬀerent sources. In this case, it is imperative to make the distinction between both
views, since change detection in time-series and drift detection in data streams with
independent examples are distinct research topics that require remarkably diﬀerent
approaches. To elucidate, consider a problem where one of the descriptive features
is a time-series deﬁned by a strictly crescent monotonic function. Any two nonoverlapping sliding windows over this series have statistically diﬀerent probability
distributions. Although this diﬀerence in the distribution exists for the time-series,
it may not be indicative of a change in other aspects of data. One example is the
analysis of the behavior of ﬁsh species. Any consistent change in water temperature
is statistically detected. However, the magnitude of this change may not be enough
to aﬀect ﬁsh behavior.
2.4 Types of Concept Drifts
In the classiﬁcation task, a predictive model learns a function that maps the input
variablesX representing the feature space to discrete output variables Y of class
labels. Fawcett and Flach (2005) state that there are two types of problems based
on the causal direction of such a relationship between the feature space and the
class labels. Additionally, only some types of drifts can occur for each type of
problem. The types of problems are:

## Page 7 / 54

Challenges in Benchmarking Stream Learning 7
X→Y : the class label is derived from the behavior of the instance. One example
is recognizing speciﬁc body movements of a person with sensors. The joint
distribution is often written as P (X, Y ) = P (Y|X)P (X);
Y→X : the class label determines the values of the features. One example is a
disease diagnosis in which the disease causes symptoms. The joint distribution
is often written as P (X, Y ) = P (X|Y )P (Y ).
Furthermore, drift incurs a diﬀerence between the two concepts. To simplify
our discussion, we call PA the probability before the drift and PB the probability
after the drift has occurred so that we can compare both distributions.
Based on the types of problems listed before, Moreno-Torres et al. (2012) review and compile the nomenclatures and deﬁnitions from literature into a single
reference list of types of drifts. Although general changes are commonly referred
to as concept drift in the data stream literature, Moreno-Torres et al. (2012) provide a normalization in which all types of change go by dataset shift. Additionally,
a change belongs to one among three more speciﬁc types: covariate shift , prior
probability shift, and concept shift.
Covariate shift refers to changes in the feature space alone and, according to
Moreno-Torres et al. (2012), only happens in X →Y problems. It is deﬁned as
follows:
Deﬁnition 1 Covariate shift is the case wherePA(Y|X) = PB(Y|X) and PA(X)̸=
PB(X).
Prior probability shift refers to changes in the class proportions alone. It is the
main subject of study in a new subﬁeld of Machine Learning called class prior
estimation or quantiﬁcation (Gonz´ alez et al., 2017). According to Moreno-Torres
et al. (2012), this type of change only happens in Y→X problems, and is deﬁned
as follows:
Deﬁnition 2 Prior probability shift is the case where PA(X|Y ) = PB(X|Y ) and
PA(Y )̸= PB(Y ).
We note that, in prior probability shift, although PA(X|Y ) = PB(X|Y ), it is
not necessarily true that PA(Y|X) = PB(Y|X). This can be easily observed by
changing P (Y ) in datasets with classes that highly overlap. To illustrate, recall
that classiﬁers typically learn to classify instances in a region of the feature space
as the most common class in the region. However, which is the most common class
is subject to change according to alterations in P (Y ). Yet, the behavior of each
class, individually, remains the same.
Concept shift is a change in the relationship between the feature space and the
class labels, and is, according to Moreno-Torres et al. (2012), the hardest type of
shift. It is deﬁned as follows:
Deﬁnition 3 Concept shift is the case where one of the following happens:
1. PA(Y|X)̸= PB(Y|X) and PA(X) = PB(X) inX→Y problems;
2. PA(X|Y )̸= PB(X|Y ) and PA(Y ) = PB(Y ) inY→X problems;
3. PA(Y|X)̸= PB(Y|X) and PA(X)̸= PB(X) inX→Y problems;
4. PA(X|Y )̸= PB(X|Y ) and PA(Y )̸= PB(Y ) inY→X problems.

## Page 8 / 54

8 V. M. A. Souza et al.
Condition 1 states that the proportions of the classes, given the characteristics
of the data points, change, while the distribution of these characteristics remains
the same. In practical terms, we have two eﬀects. First, if we ignore the label
information and compare the data before and after the drift, they have the same
probability distribution. Second, the proportion of classes in regions of the feature
space changes. The only diﬀerence between conditions 1 and 3 is that the latter is
free of the restriction of preserving P (X).
Condition 2 states that the characteristics that deﬁne each class change. However, P (Y ) must be preserved, while P (X) can change. P (Y ), on its own, dictates
the proportion of the classes considering all data points. The only diﬀerence between conditions 3 and 4 is that the latter is free of the restriction of preserving
P (Y ). For that reason, condition 4 is similar to the prior probability shift with
the aggravating factor that the characteristics of each class have changed.
Moreno-Torres et al. (2012) state that the concept shifts 3 and 4 are rarer and
possibly impossible to tackle. On the other hand, the two ﬁrst shifts are easier to
deal with. However, we ﬁnd no reason to believe that concept shifts 3 and 4 are
rare. We emphasize that such conditions are not mutually exclusive except for the
pairs (2, 4) and (1 , 3).
A simple global linear transformation that moves all instances towards some
direction can cause concept shift 3. In this situation, if the proportion of classes
is kept the same after the drift, we would also simultaneously fulﬁll condition 2.
Otherwise, we would simultaneously fulﬁll condition 4. A linear transformation
without changes in the proportion of classes is illustrated in Fig. 2. We analyze a
case of temporal overlap on real-world data in Section 5.6.
Before 
drift 
PA(c 1) PA(c 2)
PB(c 1) PB(c 2)
After 
drift 
T emporal 
overlap 
Fig. 2 Illustration of a case where PA(Y |X) ̸= PB(Y |X) and PA(X) ̸= PB(X). There are
two classes ( c1 and c2). Their distributions are shown before and after the drift happened.
The drift is a global linear transformation that moved the average feature value two units up.
The green shade illustrates a temporal overlap: instances that belong to class c1 would seem
to belong to class c2 according to the outdated distributions. In this particular example, since
there is no change in the proportion of classes, it is also true that PA(X|Y ) ̸= PB(X|Y ) and
PA(Y ) = PB(Y ). Therefore, this ﬁgure represents concepts shifts 2 and 3.
Fig. 2 has a temporal overlap. The temporal overlap is a superposition of instances that belong to diﬀerent classes in the feature space, and that only occurs
if we ignore the temporal aspect of the data. For example, if we process the whole
dataset at once. If we fail to temporally split the data so that we can separately
analyze the concepts before and after the drift, we identify a greater class overlap
than the one that exists in each concept individually. In this particular case, our

## Page 9 / 54

Challenges in Benchmarking Stream Learning 9
view of the data would suggest that around 50% of c1 overlaps with around 50% of
c2 in the feature space, which is a strikingly harder classiﬁcation scenario than the
one found in each isolated concept. The existence of temporal overlap reinforces
the importance of adequately choosing the parameters of observation windows.
While concept shifts 3 and 4 may be hard to tackle in typical batch learning
problems, some assumptions can make them identiﬁable in streams. For instance,
changes in a stream can be incremental and, therefore, traceable over time (Dyer
et al., 2014; Souza et al., 2015a,b), or can always lead to a previously seen distribution of the data (Reis et al., 2018b).
Finally, we contest the easiness of concept shift 1: in fact, this type of drift
is impossible to detect in unsupervised settings, since we can only observe P (X)
and it does not change (Zliobaite, 2010). This fact is visually illustrated in Fig. 3.
Since the proportion of the classes is changed non-uniformly to preserve P (X),
this ﬁgure also illustrates a concept shift 4.
/uni00000015
/uni00000013/uni00000015
/uni00000015
/uni00000014
/uni00000013
/uni00000014
/uni00000015
/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni00000024
/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni00000025
(a) Before concept drift
/uni00000015
/uni00000013/uni00000015
/uni00000015
/uni00000014
/uni00000013
/uni00000014
/uni00000015
/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni00000024
/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni00000025 (b) After undetectable drift
/uni00000015
/uni00000013/uni00000015
/uni00000015
/uni00000014
/uni00000013
/uni00000014
/uni00000015
/uni00000038/uni00000051/uni0000004e/uni00000051/uni00000052/uni0000005a/uni00000051/uni00000003/uni00000046/uni0000004f/uni00000044/uni00000056/uni00000056 (c) Unlabeled data
Fig. 3 Illustration of a concept drift that is undetectable without true labels in a twodimensional feature space with two classes. Red dots represent events belonging to class A,
that are generated with the red-shaded area. Blue dots represent events belonging to class B,
that are generated withing the blue-shaded area. In general, undetectable changes without true
labels are those in which PA(X) = PB(X) while PA(Y |X) ̸= PB(Y |X) (Reis et al., 2018a).
We note that, in speciﬁc settings, a drift may be undetectable when windows
are far apart from each other in the stream. However, if there are intermediate
changes of concept between the distributions estimated upon the ﬁrst and last
windows, and with proper setting of the observation windows, it may be possible
to trace the evolution of the drift over time and detect it without true labels (Dyer
et al., 2014; Souza et al., 2015a,b). Fig. 4 illustrates such a case.
Kull and Flach (2014) extend the work of Moreno-Torres et al. (2012) by
introducing graphical notations of the dataset shifts types mentioned above, and 12
new additional sub-types of shifts. We point the interested reader to this paper for
further information on this topic. Oppositely, Kelly et al. (1999); Tsymbal (2004)
oﬀer a simpliﬁed view that is often enough to specify a concept drift problem.
According to them, concept drift occurs when P (X), P(X|Y ) or P (Y|X) change.
The cases where P (X) changes, while P (Y|X) does not change, are referred to
as virtual drift. Opposite cases, where P (Y|X) changes while P (X) does not, occur
due to alterations in the hidden context. Hidden context is the information that is
not included in the observable predictive features but is relevant to determine the
class label (Harries et al., 1998). Furthermore, severe changes in P (Y ) can lead to

## Page 10 / 54

10 V. M. A. Souza et al.
t0 t1 t2 t3 t4
Fig. 4 Illustration of a case of traceable incremental drift. There are two classes, distinguishable by their unique color. This illustration presents ﬁve snapshots that represent the evolution
of the data over time. The thick arrow represents a passage from one snapshot to the following.
The curved arrow represents the movement of the class in the feature space. A brighter class
with a dashed border represents the previous position. Notice that if we only compare t0 with
t4, we have a case that approximates PA(Y |X) ̸= PB(Y |X) and PA(X) = PB(X), which is
undetectable without true labels. However, with the support of the intermediate distributions,
we can trace the geometric evolution of the data and therefore distinguish both distributions.
class imbalance. Such changes in the class distribution can make a majority class
to become a minority class and vice-versa in the course of a stream (Maletzke
et al., 2018).
Perceiving the occurrence of a concept drift can carry diﬀerent meanings and
consequences depending on the application. For instance, in an application where
there is interest in detecting new classes of data, a change where a new cluster of
data points appears may represent the emergence of a novelty. Fig. 5 illustrates
diﬀerent practical types of drift found in data stream literature. In this example,
circles represent instances, and colors represent classes. The ﬁgure also shows the
decision boundary that discriminates the classes.
Original data Real concept drift Virtual concept drift 
Class fusion Class imbalance Novel class emergence 
Fig. 5 Some types of concept drifts in data stream frequently found in literature. Dashed
lines indicate the separation margin between the classes.

## Page 11 / 54

Challenges in Benchmarking Stream Learning 11
2.5 Data Stream Realization
One point of concern is the nature of the sequence and how it translates to the
divergences in how consecutive instances are observed.
In certain sequences, the last observed instance is a transformation of previous
instances and thereforecould not have existed before their materialization (Harries,
1999; Zhu, 2010). This is the case of most data from time-series problems. In a
considerable amount of them, the observed data are complete, i.e., we observe all
instances of the problem, and the practical objective is to predict future readings
according to the data trend. In that sense, each individual observation can be
considered of little importance. We highlight that in those cases, the feature-values
registered for each observation are highly dependent on previous observations. For
that reason, these data are the most aﬀected by temporal dependence. We name
sequences under this scenario materialization sequences.
We recall that instances are distributed according to a background concept
that evolves over time. However, in certain cases, the order in which instances were
actually observed does not imply the necessary order of their materialization, that
is, the order in which the instances started to exist (Zliobaite, 2011; Ikonomovska
et al., 2011; Katakis et al., 2009; Souza et al., 2015b; Reis et al., 2016). One example
is the data collected by a mosquito trap that measures ﬂight characteristics of
insects using sensors. While multiple insects may coexist in the trap vicinity at the
same time, the ones that ﬂy into the trap do so in a somewhat randomized order.
However, all those insects have their behaviors aﬀected by shared environmental
factors, which change over time. In most cases that follow this structure, the
observed data are only a sample of a more extensive set of instances that may
not ever be observed, and the practical objective is to determine the class of each
instance. In that sense, each observation is considered of high importance. We
name sequences under the described setting observational sequences.
The arrangement of data points in a sequence is commonly tied to time, be it
the order in which data points were observed or materialized in the world. We call
sequences that have their arrangement tied to time temporal sequences. However,
not all streaming data are tied to the chronological order of events.
Therefore, another relevant aspect of data stream is the physical nature of the
sequence. Frequently, a stream is not chronological even though there is a logical
sequence (Blackard and Dean, 1999). One example is the analysis of the pavement
quality of a road (Souza, 2018; Souza et al., 2018). The extent of the road can be
split into sections that are analyzed individually, and the order of such sections
can follow their spatial positions. Therefore, the resulting sequence follows a logical
sequence, yet the actual time when data for each section were collected is irrelevant
and interchangeable. When the order of the instances is related to their spatial
disposition, we call the resulting sequence spatial sequence. Finally, sequences not
tied to time nor space are called logical sequences.
When the concept behind the instances in a sequence is tied to either time or
space, a relevant aspect is the spacing between instances. In typical materialization
and spatial sequences, instances can generally be observed at regular time/space
intervals. However, there are cases where the time/space between consecutive observations vary, and this setting poses particular complications for observational
sequences. For example, in the mosquito trap application mentioned above, we
know that diﬀerent species of ﬂying insects show more or less activity according

## Page 12 / 54

12 V. M. A. Souza et al.
to their circadian rhythm (Shinkawa et al., 1994). Fig. 6 illustrates the circadian
rhythm of Culex quinquefasciatus mosquitoes, measured by the trap’s sensor over
a week.
Time of day
# events
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 230
500
1000
1500
2000
2500
3000
3500
Fig. 6 Circadian rythm of Culex quinquefasciatus (Souza, 2016). Each bar represent the
amount of insect passages over the trap’s sensor given a time of day.
We can see that mosquitoes of this species abruptly become inactive in the
dawn and resume their activity in the dusk. Observations done by the trap rely
on the activity of insects so that it is improbable to collect data while specimens
are inactive. However, even if the trap is not making observations, time passes and
environmental condition changes, and therefore the ﬂight characteristics of insects
also changes. If the sequence does not have timestamps for each observation, we
could probably observe an abrupt and inexplicable change of behavior for Culex
quinquefasciatus. On the other hand, with timestamps, we can notice that we
lacked data for a prolonged period in which the behavior might have changed.
The example mentioned above illustrates the importance of temporal or spatial
marks to understand concept drifts. In similar cases where the order of observations
deﬁnes the sequence of the stream according to when they were made, if the
observations are not evenly distributed and are tied to the temporal or spatial
progression of the background environment, it is essential to include timestamps
or longitudinal data to understand changes.
For the sake of completeness, there are cases where data are considered to be
a stream only due to its long length, although it lacks logical ordering (Cattral
et al., 2002; Vergara et al., 2012). If an instance is as likely to be observed at timet
as at time t′, t̸= t′,∀t, t′, then there is no concept drift. In fact, any two windows
of data are going to be equally distributed, since both are uniform samples of the
data. In such cases, forgetting mechanisms to discard old data is not beneﬁcial,
but should not be unfavorable to the performance of the classiﬁer either. We call
sequences under this setting unordered sequences (Cattral et al., 2002).
2.6 Stream Classiﬁcation
Classiﬁcation is probably the most common task in data mining and a topic of
active research in data stream (Street and Kim, 2001; Bifet et al., 2013; Souza
et al., 2015b; Gomes et al., 2017). Classiﬁcation is the process of inducing a general

## Page 13 / 54

Challenges in Benchmarking Stream Learning 13
model from previously known data (training data), and then using such a model
to predict class labels (discrete values) for previously unknown data objects (test
data). Diﬀerently from batch learning, in data stream or online learning, the test
examples arrive continuously in an orderly fashion over time, and a classiﬁer should
generally predict the label of each instance in real-time or at least before the arrival
of the next example.
In classiﬁcation, the objective is to build a model that approximates the true
relation between instances (X ) and their respective class-labels (Y) and, therefore,
is capable of predicting the class-labels of unlabeled instances. In other words, we
want to build h :X →Y such that h(Xt) = yt. The task of inducing h from
labeled instances is referred to as training.
In batch learning, the classiﬁcation model h is typically induced beforehand
using a training set of labeled instances. However, in data stream classiﬁcation,
several approaches diﬀer signiﬁcantly regarding which data are used for training.
A recent approach assumes that labeled data for all possible concepts are available before the stream begins to be processed (Reis et al., 2018a,b). In that case, individual classiﬁers are trained for each concept without regard for the data stream
at ﬁrst. Such models are only later deployed to classify examples in the stream.
However, there is not a single attribute that can easily identify, which is the concept of current data. Therefore, we need means to detect, which is the adequate
classiﬁer to be used for recent data.
Nevertheless, the most common setting in the data stream community is that
there is no separate training set to induce a deﬁnitive model h. Thus, the model
needs to be constructed and updated on the ﬂy as new data are observed (Zliobaite
et al., 2015). Among approaches in this setting, some methods incrementally evolve
the classiﬁcation model but do not adapt to changes in data. Such methods are
capable of dealing with the fast pace of streaming data and keep low usage of
memory. One example is the Very Fast Decision Tree – VFDT (Domingos and
Hulten, 2000), where new incoming examples update the statistics of the leaves
of a tree-based model. As more instances are observed, the statistics are used to
decide which and when leaves are split into new leaves, resulting in the growth of
the tree. Recently, VFDT was adapted as the newly introduced Extremely Fast
Decision Tree to become faster (Manapragada et al., 2018).
However, due to changes in data distribution, the learner must incrementally
adapt its model h over time or perform updates when necessary to maintain stable
predictive performance. Therefore, we also have a sequence of models h1, . . . , hi,
which can be discarded or reused in recurring situations. There are two main
approaches to deal with concept drifts in classiﬁcation problems (Khamassi et al.,
2018): i) evolving models, and ii) adaptive models. Gama et al. (2014) names such
approaches of i) blind, and ii) informed.
The evolving models update the learner at regular intervals without considering
whether changes have occurred. To do so, the model uses mechanisms for learning
new concepts and forgetting old ones. A common approach is a sliding window
with a ﬁxed or variable length to store training examples or by weighting the data
by age/utility (Klinkenberg, 2004). Such strategies consider that the most recent
data are more representative of the current concept, so we can discard old data or
assign them less weight. The main weakness of this approach is that the forgetting
of old concepts is carried out at a constant speed for the whole time. Therefore,
old data are discarded even when changes are not happening.

## Page 14 / 54

14 V. M. A. Souza et al.
The evolving methods are naturally able to handle gradual and incremental
drifts. Examples of work that implement evolving approaches are the algorithms
from the FLORA family (Widmer and Kubat, 1996). In FLORA2, incoming examples are added to the window, and the oldest ones are deleted. A naive approach
that falls into the same category is periodically retraining a new classiﬁcation
model with the last observed instances. Besides, some algorithms recently proposed, such as COMPOSE (Dyer et al., 2014) and SCARGC (Souza et al., 2015b),
use the sliding window strategy to deal with incremental changes in scenarios in
which the actual labels of test instances are never available to the learner.
Adaptive models explicitly detect concept changes using drift detectors, updating the model only when changes are ﬂagged. One of the advantages of explicit
detection is the production of information about the dynamics of the data generation process and the reduced amount of updates in conditions without concept
changes. An example of relevant work that uses drift detection is the extension
of VFDT called Concept-adapting Very Fast Decision Trees – CVFDT (Hulten
et al., 2001). CVFDT works by keeping its model consistent with a sliding window
of examples. However, it does not need to learn a new model from scratch every
time a new example arrives; instead, it updates the suﬃcient statistics at its nodes
by incrementing the counts corresponding to the new examples and decrementing
the counts corresponding to the oldest example in the window (which now needs
to be forgotten). If the concept is changing, some splits that previously passed
the Hoeﬀding test will no longer do so, because an alternative attribute now has
higher gain. In this case, CVFDT begins to grow an alternative subtree with the
new best attribute at its root. When this alternate subtree becomes more accurate
on new data than the old one, the old subtree is replaced by the new one.
According to Khamassi et al. (2018), the main issues of the approaches that use
drift detectors are related to knowing i) how to track concept drift, ii) which data
to keep and which data to forget, and iii) how to adapt the learner parameters
and structure to react according to the requirements of the new environment.
2.7 Drift Detection
Drift or change detection is the task of identifying signiﬁcant data distribution
changes in a stream. Although drift detection is a common mechanism of adaptive
stream classiﬁers as a trigger for model updates, it is also a separated task from the
classiﬁcation process that contributes to other real applications as those related
to signal analysis or time-series.
For example, change detection can be used to provide alerts when the value of a
stock is falling in the market (Oh and Kim, 2002) or identifying a fault in the monitoring of industrial processes (Venkatasubramanian et al., 2003). An important
application of change detection methods is in burst detection (Gama, 2010). Burst
regions are time intervals in which some feature values are unexpected. For example, gamma-ray burst in astronomical data might be associated with the death of
massive stars; bursts in document streams might be valid indicators of emerging
topics, and so on.
In classiﬁcation problems, drift detection methods are categorized into two
major groups according to the availability of labeled data in the stream (Faithfull et al., 2019): i) supervised, and ii) unsupervised. Supervised drift detection

## Page 15 / 54

Challenges in Benchmarking Stream Learning 15
methods assume the immediate availability of class labels of each instance. These
methods surveil indicators of classiﬁcation performance, such as accuracy to detect drifts. On the other hand, when the class labels are delayed or unavailable,
unsupervised methods detect drifts by comparing data distributions at diﬀerent
time intervals.
Based on the taxonomy proposed by Gama et al. (2014) of dimensions which
characterize drift detection methods, we consider three main categories:
I) Methods based on diﬀerences between two distributions. In this
approach, the methods monitor the distributions of two data windows. We can
consider a reference window with old data and a detection window composed of
recent data. These windows are compared using statistical tests, with the null
hypothesis that the data of both windows are drawn from the same distribution.
Thus, a concept drift is ﬂagged when the test rejects the null hypothesis. The windows can contain unsupervised information as the raw data, learner parameters,
classiﬁer’s outputs such as probabilities estimate or classiﬁcation scores, as well
as supervised information such as the error rate of the classiﬁer or even the class
labels.
Some parameters are fundamental to the success of these methods, such as
how to measure the change and how to determine the size of the windows. To
measure the change, diﬀerent types of statistical tests as univariate or multivariate and parametric or non-parametric can be employed. Examples of tests are the
Kullback-Leibler divergence (Dasu et al., 2006), Hotelling’s t2 (Hotelling, 1992),
semi-parametric log-likelihood – SPLL (Kuncheva, 2013), Kolmogorov-Smirnov
(Reis et al., 2016). Regarding the window size, it is important to note that a window smaller than the changing rate may lead to a higher number of false negative
detections, and a window larger than the changing rate may delay the detections.
Most of the work done is based on ﬁxed-size windows, where delay in detections
is frequent (Ganti et al., 1999; Kifer et al., 2004; Dasu et al., 2006). Other pieces
of work consider windows with dynamic size. For example, ADWIN (Bifet and
Gavalda, 2007) ﬁnds two windows of diﬀerent sizes through multiple tests between consecutive sub-windows within a window with ﬁxed and large enough size.
To detect drifts, ADWIN uses the Hoeﬀding bound to compare the sub-windows.
Similarly to ADWIN, SEED (Huang et al., 2014) uses two sub-windows with dynamic sizes but also performs block compressions to reduce the number of window
comparisons. It also computes the volatility shift to describe the relationship of
proximity between consecutive drift points in the stream.
II) Methods based on sequential analysis. The method Sequential Probability Ratio Test – SPRT (Wald, 1947) is the foundation of detection methods such
as CUSUM and Page-Hinkley (Page, 1954). To better understand SPRT, consider
a subsequence of N examples from the streamDS ={X1, X2, . . . ,XN , . . .} where
the subset of instancesD1 ={X1, . . . ,Xw} with 1 < w < N is generated from an
unknown distribution PA and the subsetD2 ={Xw+1, . . . ,XN} is generated from
another unknown distribution PB. A change is declared at time w if the probability of observing examples under the distribution PB is signiﬁcantly higher than
the PA. For this veriﬁcation, SPRT tests the logarithm of the likelihood ratio considering the two distributions. The main diﬀerence compared with the approaches
previously discussed is that the test of SPRT is made sequentially one by one with
diﬀerent values of w, until the decision to accept or refuse the null hypothesis
that PA and PB are the same distribution. SPRT is a classic method proposed

## Page 16 / 54

16 V. M. A. Souza et al.
in statistics, and data stream applications still employ it to detect changes with
competitive performance (Faithfull et al., 2019).
III) Methods based on statistical process control. For decades, the
quality control of products in continuous manufacturing is made using standard
statistical techniques called control charts. Diﬀerent methods such as DDM (Gama
et al., 2004), EDDM (Baena-Garcia et al., 2006), and EWMA (Ross et al., 2012)
are based on these statistical techniques to detect changes in data stream. These
drift detection methods consider the classiﬁcation problem as a statistical process
and monitor the evolution of some performance indicators, such as the error rate,
to apply heuristics to ﬁnd points of change. For example, the method DDM considers three diﬀerent states for the classiﬁcation error evolution: i) in-control, when
the error is stable; ii) out-of-control, when the error is increased signiﬁcantly as
compared to the recent past; and iii) warning, when the error is increasing but has
not reached the out-of-control state. The method stores the data in a short-term
memory during the warning state and rebuild the classiﬁcation model with this
data when the error state is changed to out-of-control.
The method employs a set of rules considering the mean and variance of the
Binomial distribution of the classiﬁer’s errors to deﬁne the threshold of the states.
An advantage of this method is that the rate of a change can be measured according
to the number of examples or the time between the warning and out-of-control
states. In this case, short times indicate fast changes, while longer times indicate
slower changes. Inspired by DDM, the EDDM also takes into account the distance
between consecutive errors as opposed to considering only the error magnitude.
The work of Gama et al. (2014), Ditzler et al. (2015), and Khamassi et al.
(2018) provide interesting reviews about diﬀerent drift detection methods from the
literature. Also, Gon¸ calves Jr et al. (2014) performs an experimental comparison
of drift detection methods.
3 Stream Datasets from Literature
Bifet et al. (2009) note the diﬃculty of ﬁnding large real-world datasets for public
benchmarking, especially with substantial concept drift. We would like to quantify
how this diﬃculty impacts the variety of datasets used in data stream research.
Therefore, we performed a broad literature review over the last two decades. We
analyzed more than 150 papers from high-impact conferences and top-tier journals
to ﬁnd the most used datasets. Unlike batch learning, in which a few hundred
static datasets are available for evaluation, the data stream learning community
has supported their ﬁndings in approximately 15 real-world datasets. In what
follows, we describe the most popular datasets.
Electricity (Harries, 1999). This dataset probably is the most used for the tasks
of stream classiﬁcation and drift detection. The data are from the Australian
New South Wales Electricity Market. Prices are aﬀected by demand and supply,
which are assessed every ﬁve minutes. The learning task is to predict a rise or
a fall in electricity prices, given recent consumption and prices in the same and
neighboring regions. The dataset contains 45,312 instances, eight attributes,
and two class labels (UP and DOWN);
Forest Covertype (Blackard and Dean, 1999). This dataset contains information about the forest cover type of 30 × 30-meter cells obtained from the US

## Page 17 / 54

Challenges in Benchmarking Stream Learning 17
Forest Service Region 2 Resource Information System. It contains 581,012 instances, 54 attributes, and seven class labels related to diﬀerent forest cover
types.
Poker-hand (Cattral et al., 2002). Each record of this dataset is a poker hand
consisting of ﬁve playing cards drawn from a standard deck of 52. Each card
is described by two attributes (suit and rank). The dataset contains 1,025,010
instances, 11 attributes, and 10 class labels related to a possible poker hand
such as a one pair, two pairs, ﬂush, full house , among others;
Intrusion Detection or KDDCUP99 (Tavallaee et al., 2009). This dataset is
from the KDD Cup 1999 Competition. The MIT Lincoln Labs gathered such
data for nine weeks. The data consist of raw TCP dump data from a local area
network. The learning task is to build a predictive model capable of distinguishing between normal connections and intrusive connections such as DoS
(denial-of-service), R2L (unauthorized access from a remote machine), U2R
(unauthorized access to local superuser privileges), and Probing (surveillance
and other probing) attacks. The original task comprises 24 training attack
types. The full dataset has about ﬁve million connection records, but it is
usual to consider a subset with only 10% of the size;
Airlines (Ikonomovska et al., 2011). This dataset is from the Data Expo Competition 2009. The dataset consists of ﬂight arrival and departure records of
commercial ﬂights within the USA, from October 1987 to April 2008. The
learning task is to predict whether a given ﬂight will be delayed, given the information of the scheduled departure. The dataset contains 539,383 examples,
seven attributes, and two class labels (Delayed and Not delayed);
Gas Sensor Array (Vergara et al., 2012). The dataset was gathered from January 2007 to February 2011, totaling 36 months, in a gas delivery platform facility situated at the University of California, San Diego. It comprises recordings
from six distinct pure gaseous substances: Ammonia, Acetaldehyde, Acetone,
Ethylene, Ethanol, and Toluene. The dataset contains 13,910 instances, where
each instance consists of the measurements of 16 chemical sensors attached
to an array. For each instance, only one of the gases is diluted in dry-air at
a varying concentration at a time inside of a chamber with the sensor array.
The chamber where the gases are measured avoids any interference of the dynamics of the gases to the measurements. Therefore, only the presence of the
gases should induce the conductivity of the sensors. An updated version of the
dataset includes, for each instance, the concentration of the gas (RodriguezLujan et al., 2014). A discrete number of concentrations was assessed for each
gas. The amount and which concentrations were measured according to the
gas type. Drift was expected in a class due to the diﬀerence in concentrations.
However, the original dataset was not intended to be a streaming dataset, and
each instance was sampled independently from the other ones. The dataset was
originally divided into batches that do not even follow the same logical sampling order. The classiﬁcation problem is to identify which gas is measured.
Luxembourg (Zliobaite, 2011). This dataset was constructed using the European Social Survey 2002 – 2007. The task is to classify a subject concerning the
internet usage as high or low. A possible source of drift is internet usage change
over time. The dataset has 20 features (31 after transformation of categorical
variables) based on the answers to a survey questionnaire and 1,901 examples
collected over ﬁve years;

## Page 18 / 54

18 V. M. A. Souza et al.
Chess.com (Zliobaite, 2011). This dataset comprises data from chess.com portal. The data consist of game records of one player over a period from December
2007 to March 2010 comprising seven attributes such as start date of the game,
speed of the move in days, number of moves until the end of the game, type of
the game (personal, tournament, and championship), current rating, opponent
rating, and piece’s color. Each player has a rating, which changes depending
on achieved results. A possible source of drift is the fact that a player develops
skills over time, besides engaging in diﬀerent types of tournaments and competitions. The rating and the type of game determine how the system selects
an opponent. The task of this data is to predict if the player will win, lose or
draw a game;
Ozone (Dua and Graﬀ, 2017). This data consists of air measurements collected
from 1998 to 2004 at the Houston, Galveston, and Brazoria areas. The learning
task is to predict the ozone level eight hours ahead of time. The dataset has 72
attributes, 2,534 examples, and two class labels (Ozone day and Normal day).
Sensor Stream (Zhu, 2010). This dataset contains environmental information
(temperature, humidity, light, and sensor voltage) collected from 54 sensors
deployed in the Intel Berkeley Research Lab. The whole stream contains information recorded consecutively over two months (one reading every 1–3 min).
The learning task is to identify the sensor ID based on the sensor data. This
dataset contains 2,219,803 instances, ﬁve attributes, and 54 class labels;
Powersupply (Zhu, 2010). This dataset contains hourly power supply data from
an Italian electricity company. The data were collected from two sources: power
supplied from the main grid and power transformed from other grids. The
stream contains 3-year data from 1995 to 1998, and the learning task is to
predict which hour of the day (1 out of 24 possibilities) the current power
supply belongs. The argument for concept drift is that it is mainly driven by
season, weather, time of the day (e.g., morning and evening), and the diﬀerences
between working days and weekends. This dataset contains 29,928 instances,
two attributes, and 24 class labels;
Spam Assassin Corpus (Katakis et al., 2009). This dataset consists of email
messages chronologically ordered according to their date and time of arrival.
The learning task is to identify if an email contains spam or a legitimate message. In this problem, the authors consider the occurrence of abrupt and gradual drifts. For the ﬁrst case, consider that the user can inform the machine
learning system of email ﬁltering about his/her interests by marking messages
as “interesting” or “junk”. For example, a user subscribed to a mailing list
might suddenly stop to be interested in messages containing smartphone reviews just after the purchase of a device. A situation where both abrupt and
gradual concept drifts can occur is the user regaining interest in topics that he
has been previously interested in. The dataset has 9,324 examples and 97,851
attributes. There are two classes, legitimate and spam, with the ratio around
25% of spam;
Rialto Bridge Timelapse (Losing et al., 2016). This dataset was built using
images extracted from time-lapse videos captured by a webcam with a ﬁxed
position. The recordings cover 20 consecutive days during May – June 2016,
capturing ten colorful buildings next to the famous Rialto bridge in Venice.
Each captured image was segmented to cover each building and generating ten
diﬀerent instances. The classiﬁcation problem of this dataset is to identify the

## Page 19 / 54

Challenges in Benchmarking Stream Learning 19
correct building. Continuously changing weather and lighting conditions aﬀect
the data representation over time. Each one of the ten classes of this dataset
has 8,225 examples encoded in a normalized 27-dimensional RGB histogram,
totaling 82,250 examples;
Outdoor Objects (Losing et al., 2015). This dataset was built from images
recorded by a smartphone camera in a garden environment. The task is to
classify 40 diﬀerent objects such as balls, shoes, pliers, cans, among others. One
hundred pictures were taken of each object under varying lighting conditions
(sunny and cloudy), aﬀecting the color-based representation, and from diﬀerent
distances and positions. Altogether 4,000 images were recorded and arranged
in temporal order. The examples from this dataset are represented using a
normalized 21-dimensional RG-Chromaticity histogram;
Keystroke (Souza et al., 2015b). It is a subset of the larger CMU dataset (Killourhy and Maxion, 2010), where 51 users type the password “.tie5Roanl” plus
the Enter key 400 times captured in eight sessions performed in diﬀerent days.
In the Keystroke data, the typing rhythm is used to recognize four diﬀerent
users. In this classiﬁcation task, ten features are extracted from the ﬂight time
for each pressed key. The ﬂight time is the time diﬀerence between the instants
when a key is released, and the next key is pressed. This dataset contains 1,600
instances that incrementally evolve due to the users’ practice;
NOAA Weather (Ditzler and Polikar, 2013). The dataset consists of weather
measurements collected over 50 years at Bellevue, Nebraska by the National
Oceanic and Atmospheric Administration (NOAA). This dataset contains eight
features: temperature, dew point, sea-level pressure, visibility, average wind
speed, max sustained wind-speed, minimum temperature, and maximum temperature. The learning task is to determine whether it will rain or not. The
dataset contains 18,159 daily readings of which 5,698 are rain and the remaining 12,461 are no rain.
Table 1 presents a summary of the characteristics of the datasets.
T able 1 Characteristics of the main stream learning datasets available for evaluation.1 Details
about ordering are not provided; 2 Timestamps or spatial marks are not included.
Dataset Instances F eatures Classes Sequence type Ordering
Electricity 45,312 8 2 materialization temporal
Forest Covertype 581,012 54 7 observational spatial 1,2
Poker-hand 1,025,010 11 10 observational unordered
KDDCUP99 494,021 41 23 observational temporal 2
Airlines 539,383 7 2 observational temporal
Gas Sensor Array 13,910 128 6 observational logical 1
Luxembourg 1,901 30 2 observational temporal
Chess.com 534 7 3 observational temporal
Ozone 2,534 72 2 materialization temporal
Sensor Stream 2,219,803 5 54 materialization temporal
Powersupply 29,928 2 24 materialization temporal
Spam Assassin 9,324 97,851 2 observational temporal
Outdoor 4,000 21 40 observational temporal
Rialto 82,250 27 10 observational temporal
Keystroke 1,600 10 4 observational temporal 2
NOAA Weather 18,159 8 2 materialization temporal
The small number of real-world datasets publicly available impose restrictions
on comparative studies of new proposals (Krawczyk et al., 2017). Such a lack of

## Page 20 / 54

20 V. M. A. Souza et al.
benchmark data leads to the use of approaches to simulate changes in static data
or the generation of synthetic data with concept drifts.
Some common approaches to simulate changes in real data with static distribution are (Sobolewski and Wozniak, 2013):
– Switching the features. To simulate concept drifts, we can switch the values of
some features while maintaining the class labels of a set of data samples (Ramamurthy and Bhatnagar, 2007; Zliobaite and Kuncheva, 2009). For example,
given a static dataset, we ﬁrst split it into two samples. In the second sample,
the original feature 1 replaces feature 2, the original feature 2 replaces feature
3, and so on, while the last feature substitutes feature 1. The class labels of
the examples remain the same;
– Swapping classes. In this approach, we randomly pick two classes in the data
set and swap their labels (Klinkenberg and Joachims, 2000; Kuncheva and
S´ anchez, 2008);
– Joining classes. We can join two or more classes in a unique class and consider
this one as a new concept in the stream (Vreeken et al., 2007).
– Reordering the data according to a hidden feature.We can hide a feature and use
it as a shared concept for the instances. In that case, we reorder the instances to
group instances within the same concept together. Within the same concept,
instances may be drawn uniformly to remove any other possible source of
drift. If the hidden feature is nominal, the concept drifts are usually abrupt.
In the case of an ordinal hidden feature, we can simulate incremental drift.
Numeric features can be turned into ordinal features so that we can more easily
draw instances from the same concept (Reis et al., 2018b). This approach suits
observational sequences better than materialization sequences.
Some examples of synthetic data generators widely used by the community
are STAGGER (Schlimmer and Granger, 1986), SEA (Street and Kim, 2001),
Rotating Hyperplane (Hulten et al., 2001), Random RBF (Bifet et al., 2009),
LED (Breiman et al., 1984), and Waveform (Breiman et al., 1984). We can also
cite the framework proposed by Narasimhamurthy and Kuncheva (2007), the Sine,
Line, Plane, Circle, and Boolean datasets proposed by Minku et al. (2010), and
the synthetic datasets generated by Dyer et al. (2014) and Souza et al. (2015b) to
evaluate incremental changes.
The main problem of simulating drifts in real data or the use of generators
is the introduction of data bias in the experimental evaluation. Data bias is the
conscious or unconscious use of a particular set of data to conﬁrm the desired
ﬁnding, and that can lead to incorrect conclusions (Keogh and Kasetty, 2003).
4 Criticisms to Current Datasets for Stream Learning
In addition to the reduced number of publicly available real-world stream datasets,
the most used datasets have problems such as a limited number of events and a
small number of concept drifts. Unfortunately, these issues can lead to biased or
incorrect conclusions when assessing the performance of stream algorithms. In this
section, we discuss these problems and possible consequences.

## Page 21 / 54

Challenges in Benchmarking Stream Learning 21
4.1 Uncertainty about Changes
One of the main problems regarding the existing stream datasets is the uncertainty
about the presence of concept drifts. The community frequently assumes that the
performance degradation of a classiﬁer over time is evidence of changes in the data
distribution. However, concept drift is not the only cause of performance degradation, that might have other origins such as poor generalization (e.g., underﬁtting
or overﬁtting (Domingos, 2012)) and noisy data arriving along the stream.
Even for datasets with known presence of concept drifts, the type of change
(covariate, probability, or concept shift), pattern (abrupt, gradual, incremental, or
reoccurring) and the exact moment these drifts occurred are frequently unknown.
The lack of knowledge about change characteristics and when they occur can
limit the evaluation of stream algorithms. A straightforward example is the evaluation of change detection methods that use criteria such as the probability of correct
change detection, the probability of false alarms, and the lag of detection (Gama
et al., 2014). Due to the lack of annotation of drift location in real data, the analysis of methods such as EDDM (Baena-Garcia et al., 2006), appropriated for slow,
gradual changes, and EWMA (Ross et al., 2012), ﬁt to abrupt changes, is only
possible with the aid of artiﬁcial data. Finally, the use of inappropriate datasets
for the problem tackled can lead to incorrect conclusions. One example is the use
of a dataset where changes follow PA(Y|X)̸= PB(Y|X) and PA(X) = PB(X) to
evaluate unsupervised detection algorithms.
Virtually all publications that present real data make informal assumptions
regarding the existence of drift. As far as we know, Sarnelle et al. (2015) is the
ﬁrst to make an eﬀort to quantify their assumptions. Although their work is limited
to the settings where drift is given by the incremental and spatial displacement of
the classes in the feature space, such assumptions are valid for a broader number
of existing work (Dyer et al., 2014; Souza et al., 2015b). Sarnelle et al. (2015)
introduce supervised means to measure the intensity of the displacement of the
classes, its direction, and, more importantly, its unsupervised traceability.
Webb et al. (2018) further raise awareness of the importance of measuring
drift in datasets, introducing the task of concept drift mapping. Particularly, they
measure the divergence between consecutive snapshots (built with observation
windows) of the data to represent the distributions of concepts over time. The
divergence between concepts is called, in this scenario, drift magnitude. The magnitude of the drift can be individually measured for marginals (P (X) and P (Y ))
and conditionals (P (X|Y ) and P (Y|X)) to provide diﬀerent views of the data
drift, revealing more information regarding its evolution. The authors also make
comparisons between drifts on speciﬁc attributes and the total drift magnitude to
highlight the contribution of diﬀerent attributes to the drift. Although the drift
magnitude was measured with total variation distance, any dissimilarity function
that applies to distributions can be employed. Finally, we note that this work intends to provide tools to describe data, rather than mechanisms to detect drift
actively.
Goldenberg and Webb (2018) review many applicable dissimilarity functions
to verify which are good options for measuring drift magnitude. The work targets
covariate shift (changes in P (X)) explicitly, suiting the task of unsupervised drift
detection. The authors recommend Hellinger distance to measure the divergence
between distributions of univariate and low-dimensional data.

## Page 22 / 54

22 V. M. A. Souza et al.
When distributions are approximated by samples of numeric values, the use
of Hellinger distance implies the discretization of the data with histograms. Many
options have gone untested by Goldenberg and Webb (2018) and we refer the
reader to Gonz´ alez et al. (2017) for more options and to Cha and Srihari (2002)
for the particularly interesting ORD, which takes the distance between diﬀerent
bins into account. Maletzke et al. (2019) introduces SORD, a version of ORD
that exempts the discretization of numeric values to compare univariate sample
distributions, and can be seen as a particular and fast-to-compute case of the
Earth Mover’s Distance.
Another interesting aspect to know about the data is if they contain temporal
overlap, as previously discussed in Section 2.4 and illustrated in Fig. 2. When it
is absent, one can approach the classiﬁcation problem with an incremental learner
that need not implement a forgetting mechanism to discard old concepts or a
system to switch between previous models. This scenario is signiﬁcantly less challenging than problems that must deal with temporal overlap.
We suggest a naive approach to indirectly measure temporal overlap if the
concepts are known, and data from each concept can be isolated. One can build a
classiﬁer for each concept and individually test their performance on their respective concept. An additional classiﬁer should be built with data from all concepts
and tested with a test set that also contains examples from all concepts. If the
performances of the classiﬁers for individual concepts are, on average, superior to
that of a single classiﬁer that single-handles all concepts, we have evidence that
there is temporal overlap. Otherwise, we have evidence that we do not need to use
forgetting mechanisms and incremental learners.
4.2 Temporal Dependence
Nearly almost ten years after the ﬁrst evaluation on the Electricity data in the
stream setting by Gama et al. (2004) and the use of these data by several studies
(e.g., Gama et al. (2005), Baena-Garcia et al. (2006), Bifet et al. (2009), Bifet
et al. (2010b), Brzezinski and Stefanowski (2011), Chen and He (2011), Ditzler
and Polikar (2013), Demsar and Bosnic (2018), and Shao et al. (2018)), Zliobaite
(2013) pointed out the problems of this dataset related to the temporal dependence
of class labels.
Suppose we employ a naive classiﬁer that predicts the next label to be the same
as the current label. This classiﬁer will be our baseline henceforth. For instance,
if the price goes UP now, the baseline will predict that the price will go UP for
the next time step as well. If the labels were independent, such a predictor would
achieve 51% given the class proportions of this particular dataset. However, if we
test such an approach on the Electricity dataset as it is, we obtain a much higher
accuracy of 85%. Therefore, the labels are not independent, since there are long
periods of consecutive UP and long periods of consecutive DOWN labels.
Zliobaite (2013) discusses the problem of temporal dependence for the Electricity dataset; however, another two popular datasets, Forest Covertype, and Pokerhand, also have the same problem. For example, in the Forest Covertype the data
are probably organized according to the geographical location of the observations,
although the dataset does not include annotations for the position. Thus, there is a
high probability that neighboring regions are of the same class. In the Pokerhand,

## Page 23 / 54

Challenges in Benchmarking Stream Learning 23
we have a more signiﬁcant issue. The MOA’s website provides a supposedly normalized version of the dataset and a link for the original version at UCI Repository.
The issue is that, besides not really having been normalized, MOA’s version has
a diﬀerent ordering for the instances. While the original version does not present
temporal dependence and the No-Change baseline achieves 43% accuracy (which
is the expected accuracy if there is no temporal dependence, given the proportion
of the classes on this particular dataset), the same baseline achieves staggering
75% accuracy on MOA’s normalized version. We can only wonder whether this
temporal dependence was purposefully implanted into the data and why. From
now on, we will consider the MOA’s version to illustrate the eﬀects of temporal
dependence better.
Fig. 7 presents the prequential accuracy of two classiﬁers: i) Naive Bayes with
Drift Detection Method (DDM) (Gama et al., 2004), and ii) the baseline classiﬁer
No-Change that predicts the current label to the next event for the three mentioned
datasets. Given a sequential dataset, in the prequential procedure (or test-thentrain), every example is ﬁrst used for testing and then for updating the model.
0 1 02 03 04 05 0
Time, instances
50
60
70
80
90
100Accuracy %
Naive Bayes (DDM) No-Change
.103
(a) Electricity
0 5 10 15 20 25 30
Time, instances
50
60
70
80
90
100Accuracy %
Naive Bayes (DDM) No-Change
.2.104 (b) Forest Covertype
0 1 02 03 04 05 0
Time, instances
20
40
60
80
100Accuracy %
Naive Bayes (DDM) No-Change
.2.104 (c) Poker-hand
Fig. 7 Prequential accuracy of Naive Bayes with Drift Detection Method (DDM) and the
baseline classiﬁer No-Change on the Electricity, Forest Covertype, and Poker-hand datasets.
In all cases, the baseline classiﬁer surpasses the results of the classiﬁer that
detects changes and periodically updates their model. For Electricity data, the
No-Change classiﬁer shows an accuracy of 85.33%, while the Naive Bayes with
DDM achieves only 81.23%. For Forest Covertype, the baseline presents 95.07%
of accuracy, and Naive Bayes with DDM has 88.04%. In the Poker-hand dataset,
the No-Change presents an accuracy of 74.51%, and the Naive Bayes with DDM
achieves just 61.96%.
In this sense, a new proposal that uses a solution based on the temporal dependence of the examples probably will show promising results on these data.
However, such a good performance does not necessarily mean that the classiﬁer
has a good generalization power or it adapts well to changes.
Bifet et al. (2013) proposed a new evaluation measure (Kappa-Temporal) to
avoid biased conclusions. Kappa-Temporal considers the diﬀerence between the
prequential accuracy of a given classiﬁer and the accuracy achieved by the naive
classiﬁer that ever predicts the last seen class label. However, we argue that this
measure is a palliative solution to be used in the evaluation process of stream learning methods to mitigate the consequences generated by a characteristic inherent
to some datasets. Further, we add that the baseline No-Change and the measure
Kappa-Temporal are not well suited to compare with and evaluate classiﬁers that

## Page 24 / 54

24 V. M. A. Souza et al.
do not rely on true labels, like those that make use of unsupervised drift detection
methods, since the baseline and the measure depend on such unavailable piece of
information.
4.3 Data Bias
Due to the reduced number of real datasets, we frequently come across stream
evaluations that consider three, two, or even only one real dataset together with a
larger number of synthetic data. The main problem of this practice is the possibility
of data bias, as previously discussed in Section 3.
With a reduced number of datasets, we can demonstrate any ﬁndings we
wish (Keogh and Kasetty, 2003). For example, consider the use of three datasets
to compare the classiﬁcation performance of the Naive Bayes algorithm with two
diﬀerent drift detectors: DDM and CUSUM (Alippi and Roveri, 2008).
In the ﬁrst scenario (Fig. 8-a), if we consider the Forest Covertype, Gas Sensor Array, and Ozone datasets, our obtained results would suggest that DDM
outperforms CUSUM. However, if we consider a second scenario with the datasets
NOAA, KDDCup99, and Luxembourg, we would conclude that both methods perform very similarly (Fig. 8-b). On the other hand, in a third scenario where we
chose the datasets Sensor Stream, Airlines, and Poker-hand, we can conclude that
the CUSUM outperforms DDM (Fig. 8-c).
Accuracy %
70
80
90
100
Forest CovertypeGas Sensor ArrayOzone
DDM CUSUM
(a) Scenario 1
Accuracy %
60
70
80
90
100
NOAA KDDCup99Luxembourg
DDM CUSUM (b) Scenario 2
Accuracy %
50
60
70
80
90
Sensor StreamAirlines Poker-hand
DDM CUSUM (c) Scenario 3
Fig. 8 In the ﬁrst scenario (a), we consider Forest Covertype, Gas Sensor Array, and Ozone.
In the second scenario (b), we consider NOAA Weather, KDDCup99, and Luxembourg. In
the third scenario (c), we consider Sensor Stream, Airlines, and Poker-hand. According to the
evaluated datasets, our conclusions can may be biased.
The results presented in Fig. 8, allow us to state that the use of a reduced
amount of datasets and the “right” choice of them can lead to biased conclusions.
To avoid this problem, we claim by the need of a suﬃciently large benchmarking
data that covers diﬀerent properties for stream learning, as it is already usual in
batch learning.
Besides the reduced number of datasets, the typical procedure employed for
evaluating the performance of adaptive learning models could also be responsible
for leading to biased conclusions. As noted by Zliobaite (2014), the standard procedure, named Prequential (or test-then-train) (Gama et al., 2013), allows processing
a dataset only once in the ﬁxed sequential order. The positions where and how

## Page 25 / 54

Challenges in Benchmarking Stream Learning 25
changes happen remain ﬁxed; thus, a single test concludes how well a model would
adapt to this ﬁxed conﬁguration of changes. While diﬀerent learning models have
diﬀerent adaptation rates, the results on a ﬁxed test snapshot with a few changes
may not be suﬃcient to generalize how this adaptive model would perform online on a given problem. To make the evaluation more conﬁdent, Zliobaite (2014)
proposes the employment of multiple tests with variations of the original dataset.
The various tests are generated by permuting the data order in a controlled way
to preserve local distributions.
4.4 Insuﬃcient Amount of Instances
Stream predictive models that operate in changing environments have diﬀerent
requirements from the traditional batch setting. The three main requirements
are (Gama et al., 2014):
1. Detect concept drifts (and adapt if needed) as soon as possible;
2. Distinguish drifts from noisy data;
3. Operate faster than the example arrival time and use a ﬁxed amount of memory
for any storage.
Here, we call attention to the third requirement. As data stream is frequently
deﬁned as an inﬁnite sequence of examples, accommodating such data in the machine’s main memory is considered impractical or infeasible. However, this deﬁnition is inconsistent with the number of instances present in the commonly used
stream datasets.
From the 16 popular stream datasets presented in Table 1, only two of them
have more than one million examples (Poker-hand and Sensor Stream). Further,
more than half have less than 50,000 examples, an amount that can be handled
by most of batch learning algorithms. In general, these numbers of examples do
not represent a challenge to data processing and storage. One possibility is that
researchers in the community might feel challenged enough to design memoryeﬃcient algorithms. In reality, we have noticed that very few papers analyze
the memory requirements, with some exceptions, such as the system streamDMC++ (Bifet et al., 2017).
4.5 Lack of Complex Distributions
For many real-world applications such as ﬁnancial fraud detection, natural disaster
prediction, spam ﬁlter, fault monitoring, or disease diagnosis, we have an interest
in events that occur with a very low frequency. In these cases, some classes are
diﬃcult or expensive to collect. Consequently, the classes are not equally represented in the data, which leads to the problem of class imbalance or skewed class
distributions (Batista et al., 2004). Class imbalance can cause a bias towards the
majority class, and the classiﬁers may tend to misclassify minority class examples
due to the poor generalization (Wang et al., 2013).
The machine learning community has widely researched the class imbalance
problem for more than 20 years (Chawla et al., 2004). However, this issue is still
challenging and subject of intensive research in the static learning setup (Yang and

## Page 26 / 54

26 V. M. A. Souza et al.
Wu, 2006). Although class imbalance and concept drift are intimately related when
changes occur in prior probabilities P (Y ), learning with class imbalance has still
received little attention on stream learning (Hoens et al., 2012; Ghazikhani et al.,
2013; Krawczyk et al., 2017). As recently noted by Wang et al. (2018), most existing
work in stream learning focuses on the concept drift in posterior probabilities
(i.e., real concept drift or changes in P (Y|X), as discussed in Section 2.2) and
most proposed concept drift detection approaches are designed for and tested on
balanced data streams.
Diﬀerently from static learning, in data stream setting the class distribution
is not ﬁxed. Instead, the class ratio varies, and the relationship between majority
and minority classes may change over time. It becomes even more complicated in
multi-class problems.
We believe that the lack of real stream datasets with complex distributions that
contain changes in both P (Y ) and P (Y|X) (or P (Y ) and P (X)) limit the research
and evaluation of data stream research in realistic scenarios. For example, Fig. 9-a
illustrates the changes in the classes proportion in the Electricity dataset given
a window with an arbitrary size of 1,000 instances. The class distribution barely
changes over time. As this data does not contain class imbalance, an alternative is
to under-sample one of the classes as proposed in the evaluation of Learn ++.NIE
algorithm (Ditzler and Polikar, 2013). However, this practice can be interpreted
as a modiﬁcation in the real problem characteristics.
0 1 02 03 04 0
Time, instances
0
20
40
60
80
100Class distribution %
UP DOWN
.103
(a) Electricity
0 5 10 15 20
Time, instances
0
20
40
60
80
100Class distribution %
NoRain Rain
.103 (b) NOAA Weather data
Fig. 9 Changes in data distribution over time for the classes UP/DOWN from Electricity
data and Rain/NoRain from NOAA Weather data. Each bar represents the counting of 1,000
consecutive examples from the stream.
In recent work, Wang et al. (2018) proposes the use of PAKDD 2009 credit
card (Linhart et al., 2009), UDI Twitter Crawl (Li et al., 2012), and NOAA
Weather as real-world datasets to evaluate diﬀerent approaches for imbalanced
class distributions on stream learning.
The PAKDD data were collected from the private label credit card operation
of a Brazilian retail chain. The task of this problem is to identify whether the
client has a good or bad credit, where the “bad” credit is the minority class with
9,868 examples taking 19.75% of the 49,973 examples. This dataset has gradual
changes since a client with bad credit can improve their status by meeting their
ﬁnancial commitments over time. In the same way, a good client can stop paying

## Page 27 / 54

Challenges in Benchmarking Stream Learning 27
their debts, changing the status to bad. In the UDI Twitter Crawl, the task is
to predict the tweet topic. To build this imbalanced stream dataset was chosen
a subset of 8,774 examples from the original data that include 50 million tweets
posted from 2008 to 2011. Next, the problem was reduced for two classes by using
only two out of seven possible topics. As noted by the authors, the tweet topic
change can be much faster and more noticeable when compared to PAKDD 2009
data. Both datasets are important contributions. However, they present some of
the previously discussed drawbacks, such as uncertainty about changes and an
insuﬃcient amount of examples. For the last data evaluated, NOAA Weather,
although the majority class has 12,461 examples (68.62%), this ratio is almost
constant over time, as shown in Fig. 9-b. Thus, there is a need for representative
stream data with more complex distribution changes over time to evaluate the
problem of imbalanced classes better.
4.6 Streaming as an afterthought
One glaring aspect of the data that is often used to test data stream learners is that
such data are not conceptually meant to this task. The most conspicuous example
is Poker-Hand dataset. We ﬁrst note that the size of the dataset is small enough
(around 20 megabytes) so that it can be fed to batch learners. Last, and more
importantly, the nature of any variant of a Poker game inherently grants an equal
chance for every combination of cards to be drawn at any given moment. Hands
drawn from a real deck of cards are independently and identically distributed so
that the hands in a stream should not be bound to a background hidden, evolving
concept, and there should not be temporal dependence. This means that none of
the challenges that are defended to be present in streaming data actually happen
in this dataset. This is reﬂected by the original version of Poker-hand, found at
UCI Online Repository.
However, for reasons that are beyond our knowledge, the “normalized” version
distributed at MOA’s website has a diﬀerent ordering for the hands and is biased
to present temporal dependence. While this fact is not made clear on the website,
we can suspect the reason is to make the dataset more challenging and interesting
for benchmarking data stream algorithms, despite the ordering being unnatural.
We are not against the use of reordering to repurpose a dataset to benchmark
by any means. However, we call attention to two important issues: the unnatural
ordering should be explicitly explained since it is the only source of streaming
challenges and it is not present in the original data; and the only challenge is
temporal dependence, which is still one of the least interesting problems to have
in a streaming application.
Another dataset that has been repurposed for a streaming application is Gas
Sensor Array. The data collection process reassured that each example is independent, involving the use of precision equipment to set the concentration of each gas
before registering the measurements of the sensor array. Only a discrete number
of sparsely distributed concentrations were tested, and the diﬀerent gases (that
are the class labels) are never mixed together. Instead, each gas is only diluted in
dry-air for the measurement. Similarly to the normalized Poker-hand, the only aspect directly associated with stream data is the unusual ordering of the examples,
which is not well explained.

## Page 28 / 54

28 V. M. A. Souza et al.
A less blatant example of dataset not well suited for streaming problems is
Forest Covertype. The order of the examples in the dataset is likely related to their
physical position in the world, which means that consecutive examples are likely to
share characteristics and class labels. However, there is an immeasurable number of
ways of iterating over square cells in a region, each way with its own implications in
the ordering of the examples, and consequently in the temporal dependence of the
stream. Yet, how the speciﬁc ordering in the dataset was achieved is unspeciﬁed,
and the data do not contain the geolocation of the examples. It is also debatable if
a linear representation of such data is an appropriate approach for learning tasks.
5 A Real-world Streaming Application with Concept Drifts
In this paper, we introduce to the data mining community, a benchmarking dataset
with diﬀerent properties to evaluate stream classiﬁers and drift detectors. The
dataset is based on a real-world streaming application based on the use of optical
sensors to recognize ﬂying insect species in real-time.
In the last years, our research group has been working in the next generation
of electronic insect traps to selectively capture only certain species (Batista et al.,
2011; Souza et al., 2013; Chen et al., 2014; Qi et al., 2015; Silva et al., 2015). Such
smart traps use Machine Learning techniques to recognize the insects that pass in
front of the sensor. The trap selectively captures species of interest such as vectors
of mosquito-borne diseases and agricultural pests, freeing all other species and,
therefore, reducing the impact of this control device on the environment.
For this application, we cannot assume that a stationary stochastic process generates the data due to the existence of variations in environmental conditions that
can inﬂuence the behavior of the insects. For example, temperature inﬂuences the
metabolism of insects (Taylor, 1963; Villarreal et al., 2017) Also, ambient conditions such as air pressure (Chadwick and Williams, 1949) and humidity (Mellanby,
1936) can change their ﬂying behavior. For these reasons, the data measured by the
sensor suﬀers from concept drifts over time, requiring adaptive models to perform
the classiﬁcation task of insect recognition.
We present the details of the smart trap in Section 5.1 and the optical sensor
used into the core of the trap in Section 5.2. Section 5.3 details the procedures of
data collection using our sensor on changing environments. Section 5.4 presents
the predictive features extracted from the insect signals. Finally, we introduce the
proposed stream benchmark data in Section 5.5.
5.1 Smart Trap for Insects
Controlling insect pests and vector of diseases is an important task and the main focus of the active research in the last decades. Entomologists have proposed dozens
of techniques from insecticides to biological control (Medlock et al., 2012). However, these techniques can be made safer and more cost-eﬀective with the knowledge of the spatial-temporal distributions of the insects in a certain area.
Traps are the main tool for the surveillance of insect populations. For instance,
sticky traps are used in crop ﬁelds, where they are installed and collected at regular
time intervals. A human expert is required to manually classify each collected

## Page 29 / 54

Challenges in Benchmarking Stream Learning 29
individual and count the species of interest. Although sticky traps are usually
inexpensive in terms of material cost, the whole procedure is expensive since it
involves manual counting and classiﬁcation.
The main advantage of smart traps such as proposed in our research is their
capability of counting and classifying ﬂying insects in real-time without requesting
the time and cost of analysis made by experts. Also, diﬀerently from other traps,
our device deliberately does not capture non-target species, such as pollinators
insects. Fig. 10 illustrates a recent prototype of our device. The trap turns a
fan on and oﬀ at the moment it senses a mosquito near the sensor, signiﬁcantly
reducing the power consumption.
Fig. 10 Smart Trap for counting and classifying mosquitoes in real-time using optical sensor.
To classify ﬂying insects in real-time, the trap combines the optical sensor to
measure the light variation over time and a circuit board to ﬁlter and record data,
as well as to extract predictive features which are used by a supervised machine
learning classiﬁer.
5.2 Sensor to Measure Insect Flying Data
The proposed data in this paper were obtained from an optical sensor built with
low-cost components to capture information about ﬂying insects remotely. This
sensor is the core of the electronic smart trap presented in Section 5.1. Fig. 11
shows the design of the sensor.
The sensor has two parallel mirrors face-to-face. An infrared LED uses the
mirrors to create an infrared light window that is captured by a phototransistor.
The infrared light bounces back and forth between the mirrors until it reaches the
phototransistor. When a ﬂying insect crosses the light, its wings and body partially
occlude the light, causing small variations that are captured by the phototransistor
as an audio signal. The optical device is essentially deaf to any agent that does
not cross the light. This is an important feature compared to regular microphones
which are susceptible to noise.
Fig. 12-(a) shows an example of data collected by the sensor given a mosquito
crossing. That signal was collected from a female Aedes aegypti mosquito, a vector
of diseases such as dengue, chikungunya, yellow, and Zika fever. The data consist
of an audio fragment that usually lasts for a few tenths of a second.
To classify the insect species, the wing-beat frequency is one of the most relevant pieces of information that can be extracted from the signals. When the signal

## Page 30 / 54

30 V. M. A. Souza et al.
Mirrors
Infrared
LED
Infrared
phototransistor
Infrared 
light
(a) Side-view
Infrared
LED
Infrared
phototransistor
Parallel
mirror
Parallel
mirror
Infrared
light
 (b) Top-view
Fig. 11 Optical sensor to capture information about ﬂying insects. When an insect ﬂies across
the sensor, a light variation is registered by the phototransistor as an audio signal.
is represented in the frequency domain, certain properties such as the fundamental
frequency are made explicit. In the case of insect data, the fundamental frequency
is directly related to the wing-beat frequency. Beyond the fundamental frequency,
the spectrum of a signal also has harmonic components with (typically) smaller
magnitudes multiples of the fundamental frequency. The position and amplitude
of these harmonics also constitute important information to distinguish diﬀerent
insect species. Fig. 12-(b) shows both wing-beat frequency and harmonics, given
the same signal generated by a female Aedes aegypti mosquito.
5.3 Data Collection in Changing Environment
To build the insect stream datasets with concept drifts, we collect data from different species using our optical sensor in a non-stationary environment for three
months approximately. We collected data in S˜ ao Carlos, S˜ ao Paulo, Brazil (University of S˜ ao Paulo campus).
To know the true class label of each insect passage during data collection,
we build diﬀerent collector devices in which only one insect species (with many
specimens) is present inside the collector. Temperature, humidity, luminosity, and
air pressure sensors are positioned in the internal part of the collector.
The temperature has a direct inﬂuence on the measured data by the sensor
with impact in the wing-beat frequency (Taylor, 1963; Villarreal et al., 2017; Gebru et al., 2018). However, we do not ﬁnd clear evidence that humidity has any
signiﬁcant eﬀect. For example, in Fig. 13 we show the WBF versus temperature
and humidity for female Aedes aegypti mosquitoes. This plot is similar to a typical
box plot, but it also shows the kernel probability density of the data at diﬀerent
values. To collect data for this plot, we varied temperature from 24 ◦C to 34◦C
(increments of 2◦C) while keeping relative humidity constant at 70% and varied
humidity from 55% to 80% (increments of 5%) while keeping the temperature
constant at 28◦C.

## Page 31 / 54

Challenges in Benchmarking Stream Learning 31
0 2000 4000 6000 8000 10000 12000 14000 16000
Time (seconds)
-0.5
0
0.5
Amplitude
Insectbeginstofly
acrossthesensor
Insectendstofly
acrossthesensor
Background 
noise
0.12 0.25 0.37 0.5 0.62 0.75 0.87 1
(a) Signal
0 500 1000 1500 2000 2500
Frequency (Hz)
0
20
40
Magnitude
Fundamental 
frequency(402 Hz)
Harmonics
Aedes aegypti ♀
(b) Spectrum
Fig. 12 A signal generated by the optical sensor given the crossing of an Aedes aegypti
(female) through the light and the spectrum of frequencies of the same signal. From the
spectrum of frequencies, we can see the wing-beat frequency of the insect (402 Hz) according
to the fundamental frequency. Also, the location of harmonics in the spectrum is a piece of
important information for species discrimination.
Temperature (ºC)
200
400
600
800
1,000
24 26 28 30 32 34
WBF (Hz)
Humidity (%)
200
400
600
800
1,000
556065707580
WBF (Hz)
Fig. 13 Inﬂuence of temperature and humidity on the wing-beat frequency observed forAedes
aegypti (female) mosquitoes.
To collect data that contemplate a wide range of environmental variation,
but in a controlled manner, we built chambers where we can control temperature
and humidity manually using a custom circuitry. We put the collectors inside the
chambers to gather data of diﬀerent species in parallel with the same environmental
condition. In Fig. 14, we show a chamber with ﬁve data collectors inside.
We collected around one million instances for 17 diﬀerent insect species, including mosquitoes, houseﬂies, bees, and wasps. For 7 of the 17 insect species, it
was possible to collect the data separated by sex, totaling 21 class labels. For ap-

## Page 32 / 54

32 V. M. A. Souza et al.
Fig. 14 Chamber used to control temperature and humidity conditions in data collection.
proximately three months, we varied the temperature from 20◦C to 40◦C and the
humidity from 20% to 90%, considering diﬀerent combinations of both variables.
In Fig. 15, we show the distribution of the instances from diﬀerent species over
both variables. In this plot, each blue bar represents the number of insect passages
given a value for humidity and temperature. As we can see, our data collection
has contemplated a wide range of combinations, with more instances when the
humidity is around 80%. We note that the proportions of observations made for
diﬀerent combinations of humidity and temperature do not necessarily translate
to how proportionally active the insects are concerning such variables in nature.
40
0
100
Temperature (Celsius)
3080
Humidity (%)
5000
60
40
Instances
2020
10000
Fig. 15 Number of instances observed given diﬀerent values of temperature and humidity in
the data collection for all species.
As some species are less active during certain times of the day or present a
reduced lifetime, it was not possible to collect observations for all of them covering the entire range of variation in temperature and humidity. For these reasons,
we built our datasets considering a subset with three species from both sexes,
generating six class labels. We choose the following most active species:
– Aedes aegypti. Also known as the yellow fever mosquito, is a mosquito that
can spread dengue fever, chikungunya, Zika fever, Mayaro and yellow fever
viruses, and other disease agents. This mosquito originated in Africa (Mousson

## Page 33 / 54

Challenges in Benchmarking Stream Learning 33
et al., 2005), but is now found in tropical, subtropical and temperate regions
throughout the world (Eisen and Moore, 2013);
– Aedes albopictus. Also known as Asian tiger mosquito or forest mosquito, is a
species that can be currently found in temperate and tropical Asia (its area
of origin), Europe, North and South America, Africa and several locations in
the Paciﬁc and Indian Oceans (Paupy et al., 2009). It is an epidemiologically
important vector for the transmission of many viral pathogens, including yellow
fever, dengue fever, and Chikungunya fever, as well as several ﬁlarial nematodes
such as Diroﬁlaria immitis (Gratz, 2004);
– Culex quinquefasciatus. Commonly known as the southern house mosquito,
is a medium-sized mosquito found in tropical and subtropical regions of the
world. It is the vector of Wuchereria bancrofti, avian malaria, and arboviruses
including St. Louis encephalitis virus, Western equine encephalitis virus, and
West Nile virus (Bartholomay et al., 2010).
The anatomy of the three species is very similar, especially when we consider
species from the same genus ( Aedes aegypti and Aedes albopictus). This similarity
is also observed in the ﬂight of the insects and consequently in the measured data
by the sensor. Fig. 16 illustrates with photos the species present in our datasets.
(a) Aedes aegypti
 (b) Aedes albopictus
 (c) Culex quinquefasciatues
Fig. 16 Adult male mosquitoes from the species present in our datasets. All photographs
were taken by Michele M. Cutwa (Cutwa and O’Meara, 2006).
5.4 Features Extraction
As we consider a more substantial number of species, it is clear from the pigeonhole
principle (Ajtai, 1988) that to classify those species with high accuracy it is required additional features than only the wing-beat frequency (WBF). For instance,
Fig. 17 illustrates the distributions of the wing beat frequency for 15 species across
all temperatures for which we possess data. The wing-beat frequency is one of the
most distinctive attributes available for this application. In this ﬁgure, we can see
that although some species show a well-deﬁned peak in the mean values of WBF,
there is a signiﬁcant overlap among the species. In this sense, the use of only this
feature to classify the insect species can be inaccurate.
Our signal, although optical, is very similar to audio, as previously shown in
Fig. 12, and consequently high-dimensional. Thus, we employ signal processing
techniques to extract additional discriminative features from data.

## Page 34 / 54

34 V. M. A. Souza et al.
0 200 400 600 800 1000 1200 1400
Wing-beat frequency (Hz)
Apis melifera
Aedes aegypti (female)
Aedes aegypti (male)
Aedes albopictus (female)
Aedes albopictus (male)
Chrysoperla externa
Culex quinquefasciatus (female)
Culex quinquefasciatus (male)
Diatraea saccharalis (male)
Drosophila suzukii (female)
Drosophila suzukii (male)
Drosophila melonogaster
Musca domestica
Scaptotrigona bipunctata
Tetragonisca angustula
Fig. 17 Density functions that ﬁt the histograms of diﬀerent insect species.
Each audio ﬁle was pre-processed and transformed into a feature vector. We
extracted a series of features such as the wing-beat frequency, complexity measures
of the signal spectrum, statistics from temporal representation, among others.
For the benchmarking data, we provide 33 features related to the energy sum of
frequency peaks and harmonics positions.
5.5 Insect Stream Data
Given the impact of temperature on the measured data by the optical sensor leading to the occurrence of concept drifts, we built our benchmarking data based on
changes in this variable. Each temperature was measured in Celsius degrees and
rounded to the nearest integer value. Thereby, we ordered the observations of the
examples over time in the stream following diﬀerent patterns of change in temperature while hiding this variable from the dataset. We reiterate that although
we have manipulated the sequence of the examples to control the drifts, all these
changes are feasible in the real use of the sensor on dynamic environments. Additionally, for each temperature, we uniformly sampled examples that were collected
within that temperature. As a result, we eliminate all other sources of drift beside
the changes in temperature. Finally, in addition to sampling from individual temperatures, we also vary the proportion of the classes over time, to mimic natural
inﬂuences in the activity of insects: circadian rhythm, the presence of predators,
among others. We consider the following changes, which also name our datasets:
– Incremental. In this pattern, the instances are arranged so that the temperature values are incrementally increased from 20 ◦C to 40◦C over all the
stream;

## Page 35 / 54

Challenges in Benchmarking Stream Learning 35
– Abrupt. We consider ﬁve sudden change points in this pattern. The ﬁrst
instances of the stream were collected at a temperature of 30◦C, and then they
abruptly change to 20 ◦C. After a time, the temperature back to change for
values around 35◦C. Similarly, other three abrupt changes occur until the end
of the stream;
– Incremental-gradual. In this pattern, the observed temperature in the ﬁrst
instances is around 37◦C and incrementally decrease until 35◦C. For a period,
we have a gradual change where the temperature of the instances intercalates
in the values of 35◦C 23◦C until deﬁnitively change for 23◦C. In this period,
two diﬀerent concepts are active at the same time. At the end of the stream,
the temperature back to incrementally increase until 27◦C;
– Incremental-abrupt-reoccurring. This pattern provides three recurrent cycles of incremental changes where the temperature increase from 20◦C to 40◦C.
Between the end and beginning of a cycle of incremental changes, we have an
abrupt change;
– Incremental-reoccurring. In this pattern, there exist three cycles of incremental changes over time. In the ﬁrst cycle, the temperature increases from
20◦C to 40◦C. In the second cycle, the temperature decreases from 40 ◦C to
20◦C. In the end, the temperature turns to increase to 40 ◦C. Although the
stream presents two clear recurrent patterns where the values are increased,
we also can consider the cycle of decreasing temperature as recurrent, but in
an “inverse” arrival order of the instances;
– Out-of-control. In this case, we have a lack of pattern in the occurrence of
changes in the temperature. It means that is expected the arrival over time
of instances observed at any temperature. This dataset is composed of all collected data in uniformly random order. As each example is sampled uniformly
sampled at each time during the stream, this dataset must be drift-free.
Fig. 18 graphically illustrates the patterns of changes presented in our datasets.
For the ﬁrst ﬁve patterns showed in Fig. 18, we built two datasets for each one,
being the ﬁrst with balanced and the second with imbalanced class distribution.
For the last dataset (Out-of-control), we have only an imbalanced version. Thus,
we have a total of 11 diﬀerent datasets.
Fig. 19 shows the distribution over 24 class labels from the Out-of-control
dataset. In this dataset, the tet-angustula and musca are the majority classes
with 170,220 (18.81%) and 168,819 (18.65%) instances, respectively. While classes
such as psilid and cx-tarsalis-male are the minority classes with only 17 and 157
instances, respectively. Thus, this dataset has two main challenges: the lack of
a pattern to distinguish the concepts and overcome the temporal overlap and
imbalanced distribution. One additional note is that the proportions of the classes
in the dataset are subject to data collection bias and do not represent the real
proportion of the species in nature. Furthermore, we expect such proportions to
vary according to time, region, and ambient conditions.
Fig. 20 illustrates the changes in the classes proportion over time for the two
versions of the Incremental dataset. However, we note that not all balanced data
versions are as well behaved as seen in Fig. 20-(a).
All datasets have 33 features, as previously discussed in Section 5.3. Except for
the Out-of-control dataset that has 24 class labels, all other datasets have 6 class
labels related to the species Aedes aegypti (female and male), Aedes albopictus (fe-

## Page 36 / 54

36 V. M. A. Souza et al.
Time
Temperature (°C)
20
40
30
(a) Incremental
Time
Temperature (°C)
20
40
30 (b) Abrupt
Time
Temperature (°C)
20
40
30 (c) Incremental-gradual
Time
Temperature (°C)
20
40
30
(d) Incremental-abrupt-reoc.
Time
Temperature (°C)
20
40
30 (e) Incremental-reoccurring
Time
Temperature (°C)
20
40
30 (f) Out-of-control
Fig. 18 Patterns of changes given the variable temperature to build the Insect Stream Data.
Number of 
examples
0K
60K
120K
180K
Class labels
apis-melifera
aedes-aegypti-female
aedes-aegypti-male
aedes-albopictus-female
aedes-albopictus-male
anopheles-female
anopheles-male
chironomos
cotesia
crisopideo
culex-quinq-female
culex-quinq-male
diatraea-female
diatraea-male
drosophila-suzukii-female
drosophila-suzukii-male
drosophila-melanogaster
musca-domestica
psilid
scaptotrigona-bipuncta
spodoptera-frugiperda
culex-tarsalis-female
culex-tarsalis-male
tetragonisca-angustula
Fig. 19 Class distribution of Out-of-control dataset.
male and male), and Culex quinquefasciatus (female and male). The 24 class labels
from the Out-of-control dataset can be seen in Fig. 19. Besides the higher number of class labels of this data, another interesting characteristic is the emergence
of new classes over time, which allows their use in the evaluation of approaches
for novelty detection (Masud et al., 2009). As is also often the disappearance of
certain classes over time, this dataset can be useful for assessing solutions dealing
with signiﬁcant changes in P (Y ).
In Table 2, we show a description of the datasets as the number of instances
and the position of the change points. Their names can identify the patterns of
changes for each dataset.

## Page 37 / 54

Challenges in Benchmarking Stream Learning 37
0 1 02 03 04 05 0
Time, instances
0
20
40
60
80
100Class distribution %
Class1 Class2 Class3 Class4 Class5 Class6
.103
(a) Balanced
0 100 200 300 400
Time, instances
0
20
40
60
80
100Class distribution %
Class1 Class2 Class3 Class4 Class5 Class6
.103 (b) Imbalanced
Fig. 20 Changes in class proportion for the Incremental dataset considering the balanced and
imbalanced class data versions. Each bar in the plots represents the class proportions into a
window with 1,000 consecutive examples in the stream.
T able 2 Description of the Insect Stream Datasets.
Dataset Instances Change point(s)
Incremental (bal.) 57,018 Throughout all the stream
Incremental (imbal.) 452,044 Throughout all the stream
Abrupt (bal.) 52,848 14352; 19500; 33240; 38682; 39510
Abrupt (imbal.) 355,275 83859; 128651; 182320; 242883; 268380
Incremental-gradual (bal.) 24,150 14028
Incremental-gradual (imbal.) 143,323 58159
Incremental-abrupt-reoccurring (bal.) 79,986 26568; 53364
Incremental-abrupt-reoccurring (imbal.) 452,044 150683; 301365
Incremental-reoccurring (bal.) 79,986 26568; 53364
Incremental-reoccurring (imbal.) 452,044 150683; 301365
Out-of-control 905,145 Throughout all the stream
5.6 Temporal Overlap
Interesting datasets for streaming problems include diﬀerent aspects discussed in
this article: changes in the proportions of the classes over time, changes in the
distribution of the features within each class over time, and temporal overlap, i.e.,
the dynamism of the overlap depending on which concept is responsible for the
current examples.
We showed in the previous sections that the data we are providing have changes
in the distribution of the features as we vary the temperature, and also a great
deal of overlap between classes for at least the wing-beat frequency attribute. We
consider our concept to relate to the temperature directly, and, for all but one
version of the dataset, the temperature is the hidden variable that evolves while
being stable within windows in the stream. For that reason, temperature overlap
coincides with temporal overlap, and the former is the source of the latter.
A relevant question is whether there is a smaller overlap between the classes
when we consider data for each temperature value than the data with all temperatures together. If that is not the case, we may need not worry about forgetting
mechanisms to discard old data, since new concepts are likely to occupy empty
regions in the feature space, as the temperature varies and we aggregate more data
over time. However, if class overlap varies, a classiﬁcation system can potentially
beneﬁt from identifying boundaries between diﬀerent concepts and using models
speciﬁcally trained for each one of them.

## Page 38 / 54

38 V. M. A. Souza et al.
To illustrate this idea, consider a subset of the data that contains only female
Aedes aegypti and female Culex quinquefasciatus. Each temperature was measured
in Celsius degrees and rounded to the nearest integer. We sampled 2, 500 examples
from each species for each one of the following temperatures (in Celsius): 24, 26,
28, 30, 32, and 34.
female Aedes aegypti 
Wing-beat frequency (Hz) 
From 24 to 34 °C 
Only 24 °C 
Only 34 °C 
female Culex quinquefasciatus 
Fig. 21 Illustration of a case of temporal overlap. When we can discriminate the data according to the current temperature, we have smaller class overlap in the wing-beat frequency.
In Fig. 21, we visually illustrate the diﬀerence in class overlap for wing-beat
frequency. This illustration is complemented by Table 3, which presents the numerical overlap between the two classes for each temperature. The overlap when
all temperatures are considered together is 36%, while the average overlap when
each temperature is isolated is 23%. The overlaps were estimated by taking the
minimum between histograms with 100 bins.
T able 3 Values for a case of temporal overlap. When we can discriminate the data according
to the current temperature, we have smaller class overlap in the wing-beat frequency.
Temperature (°C) 24 26 28 30 32 34
Overlap (%) 29 32 28 23 19 5
Finally, to not limit ourselves to only one feature (wing-beat frequency), we
indirectly measured the eﬀect of the diﬀerence of overlaps by evaluating a classiﬁcation task. We compared the use of individual classiﬁers for each temperature
against a single classiﬁer trained with data from all temperatures. The accuracy
rates were obtained via 10-fold cross-validation and a Random Forest classiﬁer
with 200 trees. We used all 33 features from the insect dataset. Table 4 presents
the accuracy rates obtained. The single classiﬁer achieves 84% accuracy for the
whole data, while individual classiﬁers average 90%. We note that greater diﬀerences can be observed depending on which temperature is individually assessed:
some temperatures apparently suﬀer a greater deal with temporal overlap than
other ones. One example is 24°C. It is the most diﬃcult case even with an individual classiﬁer, and is also the most harmed by the use of a conjoint classiﬁer, with
a 20% diﬀerence in their accuracy.

## Page 39 / 54

Challenges in Benchmarking Stream Learning 39
T able 4 Indirect eﬀect of temporal overlap. When we can discriminate the data according to
the current temperature, we have higher accuracy for the insect data.
Temperature (°C) 24 26 28 30 32 34
Individual classiﬁers 86 87 88 89 92 98
Single classiﬁer 66 81 88 87 89 93
6 USP Data Stream Repository
Aiming to mitigate possible ﬂaws in the experimental evaluation of future proposals on stream learning due to the lack of real-world data, we provide to the
machine learning community a new public repository called USP Data Stream
Repository2. In this repository, we make available 27 datasets from diﬀerent real
problems composed by 16 data previously evaluated by other works from literature
and 11 new datasets obtained by the optical sensor for automatic insect recognition3. It is important to note that we want to feed this repository regularly with
new data from collaborative contributions.
We suggest that stream classiﬁers and drift detection algorithms should be
tested on a wide range of datasets, mainly the real ones to avoid biased conclusions.
It is a usual practice in more consolidated areas, such as machine learning in general
(Dua and Graﬀ, 2017) and time-series (Chen et al., 2015), which contributes to
the research advancement and maturity of the data stream area. At the same
time, comparisons against baseline methods such as those proposed by Bifet et al.
(2013) are also essential for the better performance analysis of new proposals. In
this direction, we include in the repository the results achieved by two simple
baselines methods for all datasets.
7 Evaluation and Discussion
Besides to provide benchmark datasets to evaluate classiﬁers and drift detection
methods, we also report the results achieved by state-of-the-art methods in our
proposed data. The availability of benchmark data accompanied by the results
achieved by methods from literature aims to make experiments from diﬀerent
researchers from the data stream community easily comparable and reproducible.
We run all experiments of stream classiﬁcation and drift detection using the
MOA framework software (Bifet et al., 2010a), which contains implementations of
several state-of-the-art methods.
7.1 Classiﬁcation
In the experimental evaluation of the classiﬁcation task, we consider two naive
baseline classiﬁers (Bifet et al., 2013): i) No-Change and ii) Majority-Class. Both
approaches do not use any input attributes and classify only using past label
information. The No-Change classiﬁer ever predicts the next class label as the
2 Available online at https://sites.google.com/view/uspdsrepository
3 The datasets are encrypted under the following password: DMKD2018

## Page 40 / 54

40 V. M. A. Souza et al.
same as last seen class label. The Majority-Class made their prediction based on
the majority class of a moving window over the stream with 1,000 instances.
In addition to the baseline classiﬁers, we also evaluate the following stream
algorithms: i) incremental Naive Bayes (NB), ii) Very Fast Decision Trees (Hulten
et al., 2001) with Naive Bayes classiﬁers at the leaves, iii) Leveraging Bagging
with 10 VFDT in the ensemble (Bifet et al., 2010b), and iv) Adaptive Random
Forest (Gomes et al., 2017). We based our choices on the eﬃciency and popularity
of the methods available for evaluation.
We consider prequential evaluation (Gama et al., 2013) over a sliding window
of 1,000 instances to evaluate the classiﬁcation performance of the algorithms. In
Table 5, we show the results achieved by the methods.
T able 5 Prequential accuracy achieved by state-of-the-art methods in the Insect Stream Data.
Dataset No-Change Maj.Class NB VFDT Lev.Bag. ARF
Inc (bal.) 16.04 11.51 47.37 45.65 61.42 64.29
Inc (imbal.) 28.23 29.76 49.30 44.92 75.13 78.94
Abrupt (bal.) 28.98 16.07 50.77 49.85 68.39 74.34
Abrupt (imbal.) 29.15 28.49 52.18 48.46 72.28 80.02
Inc-gradual (bal.) 38.43 15.76 52.32 51.85 72.51 77.92
Inc-gradual (imbal.) 30.16 29.52 57.46 53.36 73.21 79.35
Inc-abrupt-reoc (bal.) 42.39 16.65 58.55 58.39 70.91 74.95
Inc-abrupt-reoc (imbal.) 28.16 29.76 52.34 51.03 69.13 77.60
Inc-reoc (bal.) 40.46 16.66 48.77 47.83 72.30 77.13
Inc-reoc (imbal.) 28.21 29.76 52.58 55.22 69.56 77.62
Out-of-control 13.06 18.80 45.99 44.70 53.58 70.45
For all datasets, we can note that the Adaptive Random Forest (ARF) presented the best overall results, followed by the Leveraging Bagging (Lev.Bag.). For
these methods, the overall results are around 70-80% for diﬀerent patterns of drifts,
which are slightly inferior when compared with our previous evaluations on static
data with a similar feature set in the problem of insect species recognition (Souza
et al., 2013; Silva et al., 2015; Qi et al., 2015). As expected, both incremental
algorithms (VFDT and Naive Bayes), which do not consider a strategy to deal
with concept drifts explicitly, were outperformed by more powerful data stream
classiﬁers. The poor performance of baseline classiﬁers gives us empirical evidence
that undesirable characteristics such as temporal dependence and the prevalence
of majority classes are underrepresented in our data.
Table 5 results provide a general view of the classiﬁcation performance of the
algorithms. However, in data streams, we are frequently interested in seeing these
performances over time. Besides, the performance over time of some approaches,
such as the baseline classiﬁers can help to understand the changes in the data. In
this direction, we present below the individual evaluation for each dataset from
our benchmark.
Fig. 22 shows the prequential accuracy results achieved over time by the compared methods for the balanced and imbalanced versions of Incremental data.
Given the slow speed of the incremental changes, the algorithms tend to present
more stable performances without signiﬁcant accuracy increase or decrease. In
the imbalanced version, the algorithms show instabilities in well-deﬁned points,
probably due to the P (Y ) changes.
Fig. 23 shows the results over time for the two versions of Abrupt data. In the
balanced version, we can note the presence of temporal dependence in four diﬀer-

## Page 41 / 54

Challenges in Benchmarking Stream Learning 41
0 1 02 03 04 05 06 0
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
(a) Balanced
0 100 200 300 400 500
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103 (b) Imbalanced
Fig. 22 Prequential accuracy on the Incremental data.
ent points of the stream (close to the times 14,000; 19,000; 40,000; and 52,000).
However, in all cases, they are rapidly dissolved as we can note by the poor performance of No-Change classiﬁer over time.
0 1 02 03 04 05 06 0
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
(a) Balanced
0 50 100 150 200 250 300 350 400
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103 (b) Imbalanced
Fig. 23 Prequential accuracy on the Abrupt data.
Fig. 24 shows the results over time for the Incremental-gradual data. For the
balanced version, we see a drastic fall in the performances of the classiﬁers immediately before 15,000 instances from the stream, which is related to the occurrence
of gradual drift. At this period, the stream presents instances from two diﬀerent
concepts at the same time until the change for a new concept is complete.
Fig. 25 shows the results for the Incremental-abrupt-reoccurring data. In the
balanced version, we can note three diﬀerent periods where the classiﬁers achieve
accuracy peaks in their performances, followed by a signiﬁcant fall in the ﬁrst two
cases. These periods correspond to the end of a cycle of incremental changes and
the start of an abrupt change. In the imbalanced data version, the analysis of
VFDT results can help to understand the data better. In this case, we can note
in three diﬀerent periods, an incremental fall of the classiﬁer performance with
values between 80% to 40%.

## Page 42 / 54

42 V. M. A. Souza et al.
0 5 10 15 20 25
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
(a) Balanced
0 50 100 150
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103 (b) Imbalanced
Fig. 24 Prequential accuracy on the Incremental-gradual data.
0 1 02 03 04 05 06 07 08 0
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
(a) Balanced
0 100 200 300 400 500
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103 (b) Imbalanced
Fig. 25 Prequential accuracy on the Incremental-abrupt-reoccurring data.
In Fig. 26, we show the results for balanced and imbalanced versions of Incremental-reoccurring data. Although the main diﬀerence of this dataset with Incrementalabrupt-reoccurring data is the presence of abrupt changes at two diﬀerent times,
the results are very similar to those previously shown in Fig. 25. It can mean that
abrupt changes are not responsible for signiﬁcant impacts in the performances
of the algorithms, mainly when we observe recurring concepts in the stream. In
general, the algorithms have more diﬃcult to adapt to incremental changes.
Fig. 27 shows the results for the Out-of-control data. It is interesting to note
that although this dataset has a large number of class labels and undeﬁned changes
in type and number, the classiﬁers show more stable performances over time when
compared with other datasets. However, the results are limited. For example, the
best classiﬁer (ARF), shows an overall prequential accuracy around 70%.

## Page 43 / 54

Challenges in Benchmarking Stream Learning 43
0 1 02 03 04 05 06 07 08 0
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
(a) Balanced
0 100 200 300 400 500
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103 (b) Imbalanced
Fig. 26 Prequential accuracy on the Incremental-reoccurring data.
0 200 400 600 800 1000
Time, instances
0
20
40
60
80
100Prequential accuracy %
No-Change Maj.Class NB VFDT Lev. Bag. ARF
.103
Fig. 27 Prequential accuracy on the Out-of-control data.
7.2 Drift Detection
We choose representative methods for diﬀerent drift detection approaches to evaluate the performance of detectors considering our benchmark data. Speciﬁcally,
we consider the following methods:
– Sequential analysis: Page-Hinkley Test (PHT) and CUSUM (Page, 1954);
– Statistical process control: Drift Detection Method (DDM) (Gama et al., 2004)
and Exponentially Weighted Moving Average (EWMA) (Ross et al., 2012);
– Comparison of data distributions: Adaptive Windowing (ADWIN) (Bifet and
Gavalda, 2007), SEED (Huang et al., 2014), and Statistical Test of Equal Proportions (STEPD) (Nishida and Yamauchi, 2007).
Regarding the parameters of the detectors, we consider a window size with
1,000 examples and a minimum of 100 examples to detect a drift. For a fair comparison, the remaining parameters follow the default values suggested by MOA. As
all evaluated methods require a base classiﬁer, we consider the Naive Bayes for all
approaches to standardize the experimental evaluation. Thus, we can evaluate the

## Page 44 / 54

44 V. M. A. Souza et al.
performance of drift detection methods based on the prequential accuracy without
the inﬂuence of the classiﬁcation algorithm.
In Table 6, we show the overall prequential accuracy (Acc.) and the total of
changes detected (C.D.) by the diﬀerent drift detectors evaluated considering the
Insect Stream Data. For each dataset, we highlighted the best accuracy in bold.
In general, the best results are achieved by the methods ADWIN and STEPD.
T able 6 Overall prequential accuracy (Acc.) and total of changes detected (CD) by diﬀerent
drift detectors.
Dataset ADWIN PHT CUSUM DDM EWMA SEED STEPD
Acc. CD Acc. CD Acc. CD Acc. CD Acc. CD Acc. CD Acc. CD
Inc (bal.) 52.68 3 54.17 1 56.63 5 52.72 1 47.37 0 54.96 9 56.55 30
Inc (imbal.) 61.02 136 58.79 42 59.97 99 49.32 9 50.37 104 58.50 86 59.96 225
Abrupt (bal.) 62.48 7 62.14 6 64.63 8 60.36 5 65.40 90 65.73 21 66.02 28
Abrupt (imbal.) 58.72 94 59.89 37 60.70 71 56.78 9 52.23 85 60.31 68 61.52 185
Inc-gradual (bal.) 72.26 6 68.30 7 69.20 9 65.40 6 71.39 39 70.38 14 71.51 25
Inc-gradual (imbal.) 67.70 36 62.57 20 63.53 41 55.25 15 58.90 182 62.30 64 62.32 64
Inc-abrt-reoc (bal.) 63.80 22 63.25 17 65.12 25 61.35 16 66.23 114 67.90 60 68.77 61
Inc-abrt-reoc (imbal.) 59.98 157 58.51 90 59.15 120 53.13 31 51.41 76 58.79 199 60.22 297
Inc-reoc (bal.) 65.93 26 64.59 16 65.87 21 63.96 21 66.45 108 69.82 47 69.45 59
Inc-reoc (imbal.) 60.39 152 58.16 67 59.65 122 55.13 34 51.91 96 59.00 163 59.68 242
Out-of-control 49.92 237 47.22 52 48.40 155 45.75 3 45.99 0 46.86 98 48.83 444
We chose three diﬀerent cases to better analyze the drift detection task over
time. Speciﬁcally, we present the results of STEPD, a method based on the comparison of distributions composed by data of two accuracies achieved by the classiﬁer in two times: the recent one and the overall one. The balanced data versions
of Abrupt, Incremental-abrupt-reoccurring, and Incremental-reoccurring datasets
were analyzed.
In Fig. 28, we show the prequential accuracy achieved by the Naive Bayes
classiﬁer using the STEPD drift detector method on Abrupt data. In this ﬁgure, we
can see the 28 change points detected in the vertical red lines. In this data, we have
six diﬀerent concepts that occur after ﬁve abrupt changes. Diﬀerent background
colors in the ﬁgure represent the six concepts (A-F) of this data. We can note in
Fig. 28 that even during the arrival of instances from a stable concept, the method
incorrectly detects diﬀerent change points. In most cases, the model adaptations
in these points do not lead to better accuracy, except in the last changes identiﬁed
into the concepts B, D, and F. We also can note that all abrupt changes were
correctly identiﬁed.
Similarly, in Fig. 29 we present the results on Incremental-abrupt-reoccurring
data. In this data, we have two diﬀerent points with well deﬁned abrupt changes.
However, it also occurs minor incremental changes between the abrupt changes.
The gradient in the background color of the ﬁgure represents the incremental
changes. All the changes are indicated in the top view of the ﬁgure. Given the
constant occurrence of incremental changes in these data over all the stream, we
can note a high number of change points identiﬁed by the method in Fig. 29.
Speciﬁcally, STEPD identiﬁed 61 change points.
In Fig. 30, we show the results on Incremental-reoccurring data. As this data
only present incremental changes over time, it is more diﬃcult to precisely indicate
the change points in the stream. However, we show a general view of these changes

## Page 45 / 54

Challenges in Benchmarking Stream Learning 45
0 10 20 30 40 50 60
Time, instances
0
20
40
60
80
100Prequential accuracy %
STEPD Change points detected
. 103
A B C D E F
Fig. 28 Prequential accuracy of Naive Bayes classiﬁer with STEPD drift detection method
and the change points detected considering the Abrupt data (balanced). Diﬀerent background
colors represent the six concepts (A-F) of this data.
0 10 20 30 40 50 60 70 80
Time, instances
0
20
40
60
80
100Prequential accuracy %
STEPD Change points detected
.103
Incremental 
change
Reoccurring 
incremental change
Reoccurring 
incremental change
Abrupt 
change
Abrupt 
change
Fig. 29 Prequential accuracy of Naive Bayes classiﬁer with STEPD drift detection method
and the change points detected considering the Incremental-abrupt-reoccurring data (balanced). The gradient in the background color represents the incremental changes. The abrupt
changes occur between two consecutive incremental changes. All the changes are indicated in
the top view of the ﬁgure.
by the gradient in the background color of the ﬁgure. In this dataset, STEPD
identiﬁed 59 change points.

## Page 46 / 54

46 V. M. A. Souza et al.
0 10 20 30 40 50 60 70 80
Time, instances
0
20
40
60
80
100Prequential accuracy %
STEPD Change points detected
.103
Fig. 30 Prequential accuracy of Naive Bayes classiﬁer with STEPD drift detection method
and the change points detected considering the Incremental-reoccurring data (balanced). The
gradient in the background color represents the incremental changes.
8 Conclusions
In this paper, we discuss the challenges faced by the stream learning community
concerning the reduced number of real-world data and the lack of a benchmark
to evaluate adaptive classiﬁers and drift detectors. This gap leads to the use of
synthetic data accompanied by a small number of real data in the evaluation of
new proposals. The main problem of this practice is the possibility of data bias,
which can lead to incorrect conclusions about stream algorithms behavior. We
also present a review regarding the main real datasets evaluated in the literature
and discuss some weaknesses in such data as the lack of knowledge about the
type/pattern of change and when it occurs in the stream.
To mitigate some of the problems identiﬁed in the evaluation of stream methods
concerning the lack of real data, we propose the use of 11 new datasets collected
by an optical sensor that measures the ﬂying behavior of insects. This data is used
in a relevant application of public health related to the use of a Smart Trap to
attract and capture target species such as the vector of diseases. In this application, non-stationary data are generated over time in streaming fashion due to
changes in the environment, which impacts the insects’ behavior. Our proposed
data has interesting characteristics to be explored by researchers, such as diﬀerent
patterns of changes (incremental, abrupt, gradual, and reoccurring), indicators of
the presence of each change and when they occur, the presence of complex changes
in the class distribution, a signiﬁcant number of instances, among others.
Although the proposed benchmark constitutes an essential contribution to the
stream mining community, it is also important to note that such data also have
some limitations. We highlighted two of them. First, to precisely indicate the drift
points and the types of drift, we performed a manual manipulation in the original
arrival order of the examples. Also, to avoid problems such as temporal dependence, we performed a shuﬄing procedure into a window with similar examples.
In practice, such procedures do not aﬀect the meaning of the application, which
can experience the simulated changes in real environments. However, such manip-

## Page 47 / 54

Challenges in Benchmarking Stream Learning 47
ulation could be interpreted as responsible for generating data semi-real or not
entirely real. The second limitation, which most of the datasets from literature
also presents, is the lack of time-stamps. Such limitation poses some restrictions
to evaluate issues where the time is an additional constraint factor in the learning
task. For example, with the time-stamps, it is possible to verify if the classiﬁcation
model is updated at the available time between the examples’ arrival. Also, the
algorithms can take this time into consideration to perform other updates in idle
periods of the classiﬁer.
We also provide to the machine learning community a new public repository
called USP Data Stream Repository, where we make available 27 datasets from
diﬀerent real problems composed by 16 data previously evaluated by other works
from literature and 11 new datasets obtained by the optical sensor for automatic
insect recognition. In this repository, we also present the results achieved by two
baseline methods for all datasets. This repository will be regularly fed with new
data from our future works and donation.
Acknowledgements The authors would like to thank Prof. Juliano J. Corbi and their laboratory staﬀ, as well as Edi Samuel B. Mendon¸ ca and PETE Company by the support in the data
collection. This study was ﬁnanced in part by S˜ ao Paulo Research Foundation (FAPESP) in the
grant numbers #16/04986-6, #17/22896-7, and #18/05859-3, the Brazilian National Council
for Scientiﬁc and Technological Development (CNPq) in the grant number 306631/2016-4, Coordena¸ c˜ ao de Aperfei¸ coamento de Pessoal de N´ ıvel Superior - Brasil (CAPES) - Finance Code
PROEX-6909543/D, and the United States Agency for International Development (USAID,
grant AID-OAA-F-16-00072).
References
Ajtai M (1988) The complexity of the pigeonhole principle. In: Annual Symposium
on Foundations of Computer Science, pp 346–355
Alippi C, Roveri M (2008) Just-in-time adaptive classiﬁerspart i: Detecting nonstationary changes. IEEE Transactions on Neural Networks 19(7):1145–1153
Alon N, Matias Y, Szegedy M (1999) The space complexity of approximating the
frequency moments. Journal of Computer and System Sciences 58(1):137–147
Baena-Garcia M, del Campo-Avila J, Fidalgo R, Bifet A, Gavalda R, MoralesBueno R (2006) Early drift detection method. In: International workshop on
knowledge discovery from data streams, pp 77–86
Bagnall A, Lines J, Vickers W, Keogh E (2019) The uea & ucr time series classiﬁcation repository. URL www.timeseriesclassification.com
Bartholomay LC, Waterhouse RM, Mayhew GF, Campbell CL, Michel K, Zou
Z, Ramirez JL, Das S, Alvarez K, Arensburger P, et al. (2010) Pathogenomics
of culex quinquefasciatus and meta-analysis of infection responses to diverse
pathogens. Science 330(6000):88–90
Batista G, Keogh EJ, Mafra-Neto A, Rowton E (2011) Sigkdd demo: sensors and
software to allow computational entomology, an emerging application of data
mining. In: ACM SIGKDD international conference on Knowledge discovery
and data mining, pp 761–764
Batista GEAPA, Prati RC, Monard MC (2004) A study of the behavior of several
methods for balancing machine learning training data. ACM SIGKDD explorations newsletter 6(1):20–29

## Page 48 / 54

48 V. M. A. Souza et al.
Ben-David S, Blitzer J, Crammer K, Pereira F (2007) Analysis of representations
for domain adaptation. In: Advances in neural information processing systems,
pp 137–144
Bifet A (2009) Adaptive learning and mining for data streams and frequent patterns. SIGKDD Explorations Newsletter 11(1):55–56
Bifet A, Gavalda R (2007) Learning from time-changing data with adaptive windowing. In: SIAM international conference on data mining, pp 443–448
Bifet A, Holmes G, Pfahringer B, Kirkby R, Gavald` a R (2009) New ensemble
methods for evolving data streams. In: ACM SIGKDD international conference
on Knowledge discovery and data mining (KDD), pp 139–148
Bifet A, Holmes G, Kirkby R, Pfahringer B (2010a) Moa: Massive online analysis.
Journal of Machine Learning Research 11(May):1601–1604
Bifet A, Holmes G, Pfahringer B (2010b) Leveraging bagging for evolving data
streams. In: Joint European conference on machine learning and knowledge discovery in databases, pp 135–150
Bifet A, Read J, Zliobaite I, Pfahringer B, Holmes G (2013) Pitfalls in benchmarking data stream classiﬁcation and how to avoid them. In: Joint European
Conference on Machine Learning and Knowledge Discovery in Databases, pp
465–479
Bifet A, Zhang J, Fan W, He C, Zhang J, Qian J, Holmes G, Pfahringer B
(2017) Extremely fast decision tree mining for evolving data streams. In: ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining,
pp 1733–1742
Blackard JA, Dean DJ (1999) Comparative accuracies of artiﬁcial neural networks
and discriminant analysis in predicting forest cover types from cartographic
variables. Computers and electronics in agriculture 24(3):131–151
Breiman L, Friedman J, Olshen RA, Stone CJ (1984) Classiﬁcation and regression
trees. Chapman and Hall/CRC
Brzezinski D, Stefanowski J (2011) Accuracy updated ensemble for data streams
with concept drift. In: International conference on hybrid artiﬁcial intelligence
systems, pp 155–163
Cattral R, Oppacher F, Deugo D (2002) Evolutionary data mining with automatic
rule generalization. Recent Advances in Computers, Computing and Communications 1(1):296–300
Cha SH, Srihari SN (2002) On measuring the distance between histograms. Pattern
Recognition 35(6):1355–1370
Chadwick LE, Williams CM (1949) The eﬀects of atmospheric pressure and composition on the ﬂight of drosophila. The Biological Bulletin 97(2):115–137
Chaudhuri S, Motwani R, Narasayya V (1999) On random sampling over joins.
ACM SIGMOD Record 28(2):263–274
Chawla NV, Japkowicz N, Kotcz A (2004) Special issue on learning from imbalanced data sets. ACM Sigkdd Explorations Newsletter 6(1):1–6
Chen S, He H (2011) Towards incremental learning of nonstationary imbalanced data stream: a multiple selectively recursive approach. Evolving Systems
2(1):35–50
Chen Y, Why A, Batista GEAPA, Mafra-Neto A, Keogh E (2014) Flying insect
classiﬁcation with inexpensive sensors. Journal of insect behavior 27(5):657–677
Chen Y, Keogh E, Hu B, Begum N, Bagnall A, Mueen A, Batista GEAPA
(2015) The ucr time series classiﬁcation archive. www.cs.ucr.edu/~eamonn/

## Page 49 / 54

Challenges in Benchmarking Stream Learning 49
time_series_data/
Cutwa MM, O’Meara GF (2006) Photographic guide to common mosquitoes of
Florida. Florida Medical Entomology Laboratory, University of Florida
Dasu T, Krishnan S, Venkatasubramanian S, Yi K (2006) An information-theoretic
approach to detecting changes in multi-dimensional data streams. In: Symposium on the Interface of Statistics, Computing Science, and Applications
Datar M, Gionis A, Indyk P, Motwani R (2002) Maintaining stream statistics over
sliding windows. In: ACM-SIAM symposium on Discrete algorithms, pp 635–644
Demsar J, Bosnic Z (2018) Detecting concept drift in data streams using model
explanation. Expert Systems with Applications 92:546–559
Ditzler G, Polikar R (2013) Incremental learning of concept drift from streaming
imbalanced data. Transactions on knowledge and data engineering 25(10):2283–
2301
Ditzler G, Roveri M, Alippi C, Polikar R (2015) Learning in nonstationary environments: A survey. IEEE Computational Intelligence Magazine 10(4):12–25
Domingos P (2012) A few useful things to know about machine learning. Communications of the ACM 55(10):78–87
Domingos P, Hulten G (2000) Mining high-speed data streams. In: ACM SIGKDD
international conference on Knowledge discovery and data mining, pp 71–80
Dua D, Graﬀ C (2017) UCI machine learning repository. URL http://archive.
ics.uci.edu/ml
Dyer KB, Capo R, Polikar R (2014) Compose: A semisupervised learning framework for initially labeled nonstationary streaming data. IEEE transactions on
neural networks and learning systems 25(1):12–26
Eisen L, Moore CG (2013) Aedes (stegomyia) aegypti in the continental united
states: a vector at the cool margin of its geographic range. Journal of medical
entomology 50(3):467–478
Faithfull WJ, Rodr´ ıguez JJ, Kuncheva LI (2019) Combining univariate approaches
for ensemble change detection in multivariate data. Information Fusion 45:202–
214
Fawcett T, Flach PA (2005) A response to webb and tings on the application of roc
analysis to predict classiﬁcation performance under varying class distributions.
Machine Learning 58(1):33–38
Gama J (2010) Knowledge discovery from data streams. Chapman and Hall/CRC
Gama J, Gaber M (2007) Learning from data streams: processing techniques in
sensor networks. Springer-Verlag Berlin Heidelberg
Gama J, Medas P, Castillo G, Rodrigues P (2004) Learning with drift detection.
In: Brazilian symposium on artiﬁcial intelligence, pp 286–295
Gama J, Medas P, Rodrigues P (2005) Learning decision trees from dynamic data
streams. In: ACM symposium on Applied computing, pp 573–577
Gama J, Sebasti˜ ao R, Rodrigues PP (2013) On evaluating stream learning algorithms. Machine learning 90(3):317–346
Gama J, Zliobaite I, Bifet A, Pechenizkiy M, Bouchachia A (2014) A survey on
concept drift adaptation. ACM computing surveys 46(4):44
Ganti V, Gehrke J, Ramakrishnan R (1999) A framework for measuring changes
in data characteristics. In: ACM SIGMOD-SIGACT-SIGART symposium on
Principles of database systems (PDS), pp 126–137
Gebru A, Jansson S, Ignell R, Kirkeby C, Prangsma JC, Brydegaard M (2018)
Multiband modulation spectroscopy for the determination of sex and species of

## Page 50 / 54

50 V. M. A. Souza et al.
mosquitoes in ﬂight. Journal of biophotonics p e201800014
Ghazikhani A, Monseﬁ R, Yazdi HS (2013) Recursive least square perceptron
model for non-stationary and imbalanced data stream classiﬁcation. Evolving
Systems 4(2):119–131
Gilbert AC, Guha S, Indyk P, Kotidis Y, Muthukrishnan S, Strauss MJ (2002)
Fast, small-space algorithms for approximate histogram maintenance. In: ACM
symposium on Theory of computing, pp 389–398
Goldenberg I, Webb GI (2018) Survey of distance measures for quantifying concept
drift and shift in numeric data. Knowledge and Information Systems pp 1–25
Gomes HM, Bifet A, Read J, Barddal JP, Enembreck F, Pfharinger B, Holmes
G, Abdessalem T (2017) Adaptive random forests for evolving data stream
classiﬁcation. Machine Learning 106(9-10):1469–1495
Gon¸ calves Jr PM, de Carvalho Santos SG, Barros RS, Vieira DC (2014) A comparative study on concept drift detectors. Expert Systems with Applications
41(18):8144–8156
Gonz´ alez P, Casta˜ no A, Chawla NV, Coz JJD (2017) A review on quantiﬁcation
learning. ACM Computing Surveys 50(5):74
Gratz N (2004) Critical review of the vector status of aedes albopictus. Medical
and veterinary entomology 18(3):215–227
Harries M (1999) Splice-2 comparative evaluation: Electricity pricing. Tech. Rep. 1,
University of New South Wales, Sydney, Australia
Harries MB, Sammut C, Horn K (1998) Extracting hidden context. Machine learning 32(2):101–126
Hoens TR, Polikar R, Chawla NV (2012) Learning from streaming data with concept drift and imbalance: an overview. Progress in Artiﬁcial Intelligence 1(1):89–
101
Hotelling H (1992) The generalization of students ratio. In: Breakthroughs in
statistics, Springer, pp 54–65
Huang DTJ, Koh YS, Dobbie G, Pears R (2014) Detecting volatility shift in data
streams. In: IEEE International Conference on Data Mining (ICDM), pp 863–
868
Hulten G, Spencer L, Domingos P (2001) Mining time-changing data streams.
In: ACM SIGKDD international conference on Knowledge discovery and data
mining (KDD), pp 97–106
Ikonomovska E, Gama J, Dveroski S (2011) Learning model trees from evolving
data streams. Data mining and knowledge discovery 23(1):128–168
Katakis I, Tsoumakas G, Banos E, Bassiliades N, Vlahavas I (2009) An adaptive personalized news dissemination system. Journal of Intelligent Information
Systems 32(2):191–212
Kelly MG, Hand DJ, Adams NM (1999) The impact of changing populations on
classiﬁer performance. In: ACM SIGKDD international conference on Knowledge discovery and data mining, pp 367–371
Keogh E, Kasetty S (2003) On the need for time series data mining benchmarks:
a survey and empirical demonstration. Data Mining and knowledge discovery
7(4):349–371
Khamassi I, Sayed-Mouchaweh M, Hammami M, Ghedira K (2018) Discussion and
review on evolving data streams and concept drift adapting. Evolving systems
9(1):1–23

## Page 51 / 54

Challenges in Benchmarking Stream Learning 51
Kifer D, Ben-David S, Gehrke J (2004) Detecting change in data streams. In:
International Conference on Very Large Data Bases (VLDB), pp 180–191
Killourhy K, Maxion R (2010) Why did my detector do that?! In: International
Workshop on Recent Advances in Intrusion Detection, pp 256–276
Klinkenberg R (2004) Learning drifting concepts: Example selection vs. example
weighting. Intelligent data analysis 8(3):281–300
Klinkenberg R, Joachims T (2000) Detecting concept drift with support vector
machines. In: International Conference on Machine Learning (ICML), pp 487–
494
Krawczyk B, Minku LL, Gama J, Stefanowski J, Wo´ zniak M (2017) Ensemble
learning for data stream analysis: A survey. Information Fusion 37:132–156
Kull M, Flach P (2014) Patterns of dataset shift. In: First International Workshop
on Learning over Multiple Contexts at ECML-PKDD, pp 1–10
Kuncheva LI (2013) Change detection in streaming multivariate data using likelihood detectors. IEEE Transactions on Knowledge and Data Engineering
25(5):1175–1180
Kuncheva LI, S´ anchez JS (2008) Nearest neighbour classiﬁers for streaming data
with delayed labelling. In: IEEE International Conference on Data Mining
(ICDM), pp 869–874
Li R, Wang S, Deng H, Wang R, Chang KCC (2012) Towards social user proﬁling:
uniﬁed and discriminative inﬂuence model for inferring home locations. In: ACM
SIGKDD international conference on Knowledge discovery and data mining, pp
1023–1031
Linhart C, Harari G, Abramovich S, Buchris A (2009) Pakdd data mining competition 2009: New ways of using known methods. In: Paciﬁc-Asia Conference
on Knowledge Discovery and Data Mining, pp 99–105
Losing V, Hammer B, Wersing H (2015) Interactive online learning for obstacle
classiﬁcation on a mobile robot. In: International Joint Conference on Neural
Networks, pp 1–8
Losing V, Hammer B, Wersing H (2016) Knn classiﬁer with self adjusting memory
for heterogeneous concept drift. In: IEEE International Conference on Data
Mining, pp 291–300
Maletzke A, Reis DM, Cherman E, Batista GEAPA (2018) On the need of class
ratio insensitive drift tests for data streams. In: International Workshop on
Learning with Imbalanced Domains: Theory and Applications, pp 110–124
Maletzke A, Reis DM, Cherman E, Batista GEAPA (2019) Dys: a framework for
mixture models in quantiﬁcation. In: AAAI Conference on Artiﬁcial Intelligence,
pp 1–9
Manapragada C, Webb GI, Salehi M (2018) Extremely fast decision tree. In: ACM
SIGKDD International Conference on Knowledge Discovery & Data Mining, pp
1953–1962
Masud MM, Gao J, Khan L, Han J, Thuraisingham B (2009) Integrating novel
class detection with classiﬁcation for concept-drifting data streams. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases
(ECML), pp 79–94
Matias Y, Vitter J, Wang M (2000) Dynamic maintenance of wavelet-based histograms. In: International Conference on Very Large Data Bases, pp 101–110
Medlock JM, Hansford KM, Schaﬀner F, Versteirt V, Hendrickx G, Zeller H, Bortel
WV (2012) A review of the invasive mosquitoes in europe: ecology, public health

## Page 52 / 54

52 V. M. A. Souza et al.
risks, and control options. Vector-borne and zoonotic diseases 12(6):435–447
Mellanby K (1936) Humidity and insect metabolism. Nature 138:124–125
Minku LL, White AP, Yao X (2010) The impact of diversity on online ensemble
learning in the presence of concept drift. IEEE Transactions on knowledge and
Data Engineering 22(5):730–742
Moreno-Torres JG, Raeder T, Alaiz-Rodr´ ıGuez R, Chawla NV, Herrera F (2012)
A unifying view on dataset shift in classiﬁcation. Pattern Recognition 45(1):521–
530
Mousson L, Dauga C, Garrigues T, Schaﬀner F, Vazeille M, Failloux AB (2005)
Phylogeography of aedes (stegomyia) aegypti (l.) and aedes (stegomyia) albopictus (skuse)(diptera: Culicidae) based on mitochondrial dna variations. Genetics
Research 86(1):1–11
Narasimhamurthy AM, Kuncheva LI (2007) A framework for generating data to
simulate changing environments. In: International Multi-Conference: Artiﬁcial
Intelligence and Applications (IASTED), pp 384–389
Nishida K, Yamauchi K (2007) Detecting concept drift using statistical testing.
In: International conference on discovery science, pp 264–269
Oh KJ, Kim Kj (2002) Analyzing stock market tick data using piecewise nonlinear
model. Expert Systems with Applications 22(3):249–255
Page ES (1954) Continuous inspection schemes. Biometrika 41(1/2):100–115
Pan SJ, Yang Q (2009) A survey on transfer learning. IEEE Transactions on
knowledge and data engineering 22(10):1345–1359
Paupy C, Delatte H, Bagny L, Corbel V, Fontenille D (2009) Aedes albopictus,
an arbovirus vector: from the darkness to the light. Microbes and Infection
11(14-15):1177–1185
Qi Y, Cinar GT, Souza VMA, Batista GEAPA, Wang Y, Principe JC (2015) Eﬀective insect recognition using a stacked autoencoder with maximum correntropy
criterion. In: International Joint Conference on Neural Networks, pp 1–7
Quionero-Candela J, Sugiyama M, Schwaighofer A, Lawrence ND (2009) Dataset
shift in machine learning. The MIT Press
Ramamurthy S, Bhatnagar R (2007) Tracking recurrent concept drift in streaming data using ensemble classiﬁers. In: International Conference on Machine
Learning and Applications (ICMLA), pp 404–409
Reis Dd, Maletzke A, Batista GEAPA (2018a) Unsupervised context switch for
classiﬁcation tasks on data streams with recurrent concepts. In: ACM Symposium On Applied Computing, pp 518–524
Reis DM, Flach P, Matwin S, Batista GEAPA (2016) Fast unsupervised online
drift detection using incremental kolmogorov-smirnov test. In: ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD), pp
1545–1554
Reis DM, Maletzke A, Silva DF, Batista G (2018b) Classifying and counting with
recurrent contexts. In: ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining (KDD), pp 1983–1992
Rodriguez-Lujan I, Fonollosa J, Vergara A, Homer M, Huerta R (2014) On the
calibration of sensor arrays for pattern recognition using the minimal number
of experiments. Chemometrics and Intelligent Laboratory Systems 130:123–134
Ross GJ, Adams NM, Tasoulis DK, Hand DJ (2012) Exponentially weighted
moving average charts for detecting concept drift. Pattern recognition letters
33(2):191–198

## Page 53 / 54

Challenges in Benchmarking Stream Learning 53
Saenko K, Kulis B, Fritz M, Darrell T (2010) Adapting visual category models to
new domains. In: European conference on computer vision, Springer, pp 213–226
Sarnelle J, Sanchez A, Capo R, Haas J, Polikar R (2015) Quantifying the limited
and gradual concept drift assumption. In: International Joint Conference on
Neural Networks, pp 1–8
Schlimmer JC, Granger RH (1986) Incremental learning from noisy data. Machine
learning 1(3):317–354
Shao J, Huang F, Yang Q, Luo G (2018) Robust prototype-based learning on data
streams. IEEE Transactions on Knowledge and Data Engineering 30(5):978–991
Shinkawa Y, Takeda SI, Tomioka K, Matsumoto A, Oda T, Chiba Y (1994) Variability in circadian activity patterns within the culex pipiens complex (diptera:
Culicidae). Journal of medical entomology 31(1):49–56
Silva DF, Souza VMA, Ellis DPW, Keogh EJ, Batista G (2015) Exploring low cost
laser sensors to identify ﬂying insect species. Journal of Intelligent & Robotic
Systems 80(1):313–330
Sobolewski P, Wozniak M (2013) Concept drift detection and model selection with
simulated recurrence and ensembles of statistical detectors. Journal of Universal
Computer Science 19(4):462–483
Souza VMA (2016) Classiﬁcation of non-stationary data stream with application
in sensors for insect identiﬁcation. PhD thesis, University of S˜ ao Paulo
Souza VMA (2018) Asphalt pavement classiﬁcation using smartphone accelerometer and complexity invariant distance. Engineering Applications of Artiﬁcial
Intelligence 74:198–211
Souza VMA, Silva DF, Batista G (2013) Classiﬁcation of data streams applied
to insect recognition: Initial results. In: Brazilian Conference on Intelligent Systems, pp 76–81
Souza VMA, Silva DF, Batista GEAPA, Gama J (2015a) Classiﬁcation of evolving data streams with inﬁnitely delayed labels. In: International Conference on
Machine Learning and Applications (ICMLA), pp 214–219
Souza VMA, Silva DF, Gama J, Batista GEAPA (2015b) Data stream classiﬁcation guided by clustering on nonstationary environments and extreme veriﬁcation latency. In: SIAM International Conference on Data Mining (SDM), pp
873–881
Souza VMA, Giusti R, Batista AJL (2018) Asfault: A low-cost system to evaluate pavement conditions in real-time using smartphones and machine learning.
Pervasive and Mobile Computing 51:121–137
Street WN, Kim YS (2001) A streaming ensemble algorithm (sea) for large-scale
classiﬁcation. In: ACM SIGKDD international conference on Knowledge discovery and data mining (KDD), pp 377–382
Tavallaee M, Bagheri E, Lu W, Ghorbani AA (2009) A detailed analysis of the
kdd cup 99 data set. In: IEEE Symposium on Computational Intelligence for
Security and Defense Applications (CISDA), pp 1–6
Taylor LR (1963) Analysis of the eﬀect of temperature on insects in ﬂight. Journal
of Animal Ecology 32(1):99–117
Tsymbal A (2004) The problem of concept drift: deﬁnitions and related work.
Computer Science Department, Trinity College Dublin pp 1–7
Venkatasubramanian V, Rengaswamy R, Kavuri SN, Yin K (2003) A review of
process fault detection and diagnosis: Part iii: Process history based methods.
Computers & chemical engineering 27(3):327–346

## Page 54 / 54

54 V. M. A. Souza et al.
Vergara A, Vembu S, Ayhan T, Ryan MA, Homer ML, Huerta R (2012) Chemical
gas sensor drift compensation using classiﬁer ensembles. Sensors and Actuators
B: Chemical 166:320–329
Villarreal SM, Winokur O, Harrington L (2017) The impact of temperature and
body size on fundamental ﬂight tone variation in the mosquito vector aedes
aegypti (diptera: Culicidae): Implications for acoustic lures. Journal of medical
entomology 54(5):1116–1121
Vreeken J, Van Leeuwen M, Siebes A (2007) Characterising the diﬀerence. In:
ACM SIGKDD international conference on Knowledge discovery and data mining (KDD), pp 765–774
Wald A (1947) Sequential analysis. John Wiley and Sons, Inc
Wang S, Minku LL, Yao X (2013) A learning framework for online class imbalance
learning. In: IEEE Symposium on Computational Intelligence and Ensemble
Learning, pp 36–45
Wang S, Minku LL, Yao X (2018) A systematic study of online class imbalance
learning with concept drift. IEEE Transactions on Neural Networks and Learning Systems pp 1–18
Webb GI, Lee LK, Goethals B, Petitjean F (2018) Analyzing concept drift and
shift from sample data. Data Mining and Knowledge Discovery 32(5):1179–1199
Widmer G, Kubat M (1996) Learning in the presence of concept drift and hidden
contexts. Machine learning 23(1):69–101
Yang Q, Wu X (2006) 10 challenging problems in data mining research. International Journal of Information Technology & Decision Making 5(04):597–604
Zhu X (2010) Stream data mining repository. URL www.cse.fau.edu/~xqzhu/
stream.html
Zliobaite I (2010) Change with delayed labeling: when is it detectable? In:
ICDMW, IEEE, pp 843–850
Zliobaite I (2011) Combining similarity in time and space for training set formation
under concept drift. Intelligent Data Analysis 15(4):589–611
Zliobaite I (2013) How good is the electricity benchmark for evaluating concept
drift adaptation. arXiv preprint arXiv:13013524
Zliobaite I (2014) Controlled permutations for testing adaptive learning models.
Knowledge and information systems 39(3):565–578
Zliobaite I, Kuncheva LI (2009) Determining the training window for small sample
size classiﬁcation with concept drift. In: IEEE International Conference on Data
Mining Workshops (ICDMW), pp 447–452
Zliobaite I, Bifet A, Read J, Pfahringer B, Holmes G (2015) Evaluation methods
and decision theory for classiﬁcation of streaming data with temporal dependence. Machine Learning 98(3):455–482
