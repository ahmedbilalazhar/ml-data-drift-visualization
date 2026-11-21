# Reviewed author full text

Accessed: 2026-11-05
arXiv version: 2602.02229v2
Publication eligibility is established by the PMLR record cited in the submission; this author version supplies methodological evidence.

Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts
Guangyi Zhang, Yunlong Cai, Guanding Yu, Osvaldo Simeone
Abstract
We study the problem of monitoring model performance in dynamic environments where labeled data are limited. To this end, we propose prediction-powered risk monitoring (PPRM), a semi-supervised risk-monitoring approach based on prediction-powered inference (PPI). PPRM constructs anytime-valid lower bounds on the running risk by combining synthetic labels with a small set of true labels. Harmful shifts are detected via a threshold-based comparison with an upper bound on the nominal risk, satisfying assumption-free finite-sample guarantees on the type-I error. We demonstrate the effectiveness of PPRM through extensive experiments on image classification, large language model (LLM), and telecommunications monitoring tasks.
Keywords:
Machine Learning, ICML
1
Introduction
Modern machine learning models have been widely deployed in dynamic real-world environments where the underlying data distribution may shift over time
(Zawalski et al., 2025)
. In such scenarios, systems can experience performance degradation due to factors such as changed environmental conditions
(Lipton et al., 2018)
, as well as evolving user preferences
(Guan et al., 2025)
.
This raises concerns about reliability and safety, particularly in critical applications such as autonomous driving
(Sun et al., 2025)
, medical diagnosis
(Zeb et al., 2024)
, and engineering
(Simeone et al., 2026)
. Consequently, there is a pressing need to continuously monitor the performance of deployed models.
Once a monitoring system raises an alarm, the deployed machine learning model may be deemed to require retraining or substitution. Given the cost associated with such extreme measures, recent works have suggested designing sequential testing frameworks that offer formal type-I error guarantees
(Podkopaev and Ramdas, 2022; Amoukou et al., 2024; Schirmer et al., 2025)
. As shown in Figure
, these methods aim to raise an alarm when the time-averaged accumulated expected risk exceeds a predefined threshold. The initial work
(Podkopaev and Ramdas, 2022)
introduced
supervised risk monitoring
(SRM), which is based on time-uniform concentration bounds obtained from labeled calibration data. Thanks to the anytime-valid properties of such bounds, SRM guarantees assumption-free type-I error requirements.
However, labeled data from deployed models are often scarce and costly to obtain, making conventional fully supervised monitoring challenging. Based on this observation,
Amoukou et al.
introduced purely unsupervised monitoring methodologies, which do not require labels for monitoring. However, in order to provide type-I error guarantees, purely unsupervised methods require assumptions about the calibration of the deployed model, which is used to impute missing labels. These assumptions are difficult to validate. Furthermore, unsupervised approaches typically require a large number of samples to achieve reliable detection, which may not be practical in many real-world settings.
To address these challenges, we propose
prediction-powered risk monitoring
(PPRM), a framework that generalizes SRM to the semi-supervised setting based on
prediction-powered inference
(PPI)
(Angelopoulos et al., 2023b)
. As illustrated in Figure
, PPRM assumes the availability of both labeled data and unlabeled data at each monitoring round. Following PPI, PPRM utilizes predictions on unlabeled data as synthetic labels, while correcting the resulting bias using a small set of labeled samples. In this way, PPRM can provide risk estimates with lower variance, and thus produces tighter confidence bounds than SRM. Therefore, as shown at the bottom of Figure
, the lower bound can exceed the threshold earlier, while also providing a similar type-I error guarantee.
We illustrate our procedure in three tasks, including image classification, large language model (LLM) monitoring via an LLM-as-a-judge
(Zheng et al., 2023)
, and a channel equalization task in a telecommunication system
(Zecchin et al., 2023)
.
Our main contributions are as follows:
•
We introduce PPRM, a semi-supervised extension of SRM that combines synthetic labels from unlabeled data with a small number of true labels through PPI.
•
We prove that PPRM provides rigorous assumption-free statistical guarantees on the type-I error.
•
We further develop an online adaptive procedure to adjust the reliance on unlabeled data, improving efficiency while preserving statistical validity.
•
We validate our methods through multiple case studies, including LLM monitoring via an LLM-as-a-judge, showing that PPRM achieves more accurate and timely risk monitoring compared to existing supervised and unsupervised methods.
2
Problem Formulation
In this section, we formalize the problem of monitoring model performance over time with the aim of identifying harmful distribution shifts that cause an excessive degradation in performance. We also review the benchmark approach introduced in
(Podkopaev and Ramdas, 2022)
.
2.1
Setting and Problem Formulation
As illustrated in Figure
, we focus on a standard multi-class classification framework, where the input domain is represented by a set
\mathcal{X}\subseteq\mathbb{R}^{D}
and the output label space is
\mathcal{Y}=\{1,\dots,C\}
, with
C
denoting the total number of classes.
We study the problem of monitoring the performance of a deployed model
f({x})\in\mathcal{O}
, while processing test data sequentially. The output space
\mathcal{O}
refers to the label space
\mathcal{Y}
or to the space of probability distributions over
\mathcal{Y}
, depending on whether the prediction is deterministic or probabilistic. We fix a loss function
\ell:\mathcal{O}×\mathcal{Y}\rightarrow\mathbb{R}
, which is assumed to be bounded in the interval
[0,1]
.
Each data point
({x},{y})
corresponds to an independent realization from an unknown joint probability distribution over the set
\mathcal{X}×\mathcal{Y}
.
Prior
to deployment, we assume access to a
calibration dataset
, whose data points are assumed to be sampled independently and identically from a fixed
source distribution
P_{0}
. This distribution represents nominal conditions under which the deployed model
f({x})
is known to perform satisfactorily. For instance, for a robotics application, calibration data may be collected in environments for which the robot has been extensively trained and validated.
At
deployment time
, test data are received
sequentially
over a discrete time index
t=1,2,...
, and are assumed to be sampled from a potentially non-stationary
test
distribution, denoted by
P_{t}
for
t=1,2,..
. No specific assumptions are imposed on the distributional shifts encoded by the sequence
\{P_{t}\}_{t\geq 1}
over time. The goal is to quickly detect if the distributions
P_{t}
are sufficiently distinct from the nominal distribution
P_{0}
to cause a harmful shift, translating into an excessive degradation in model performance. Throughout this paper, the notation
(x_{t},y_{t})\sim P_{t}
represents a generic input-output pair drawn from the distribution
P_{t}
, whereas the pair
(x_{t,i},y_{t,i})
corresponds to the
i
-th observed realization in the dataset
\mathcal{D}_{t}
.
In more detail, the calibration dataset
\mathcal{D}_{0}=\{({x}_{0,i},y_{0,i})\}_{i=1}^{n_{0}}
is assumed to be drawn i.i.d. from the source distribution
P_{0}
. Furthermore, at each discrete test time
t
, a dataset
\mathcal{D}_{t}=\{({x}_{t,i},y_{t,i})\}_{i=1}^{n_{t}}
is obtained whose samples
({x}_{t,i},y_{t,i})
are drawn i.i.d. from the test distribution
P_{t}
.
The expected loss of the model
f(\cdot)
at time
t
is given by
R_{t}=\mathbb{E}\left[\ell\big(f({x}_{t}),{y}_{t}\big)\right],
(1)
with
({x}_{t},{y}_{t})\sim P_{t}
, for
t=0,1,..
.
The quantity
R_{0}
corresponds to the
source risk
, i.e., to the expected loss when inputs are drawn from the source distribution
P_{0}
. As mentioned, this serves as a benchmark nominal risk, and excessive positive deviations from it signal a harmful distribution shift.
The test performance is evaluated by the
running test risk
(Podkopaev and Ramdas, 2022)
\bar{R}_{t}=\frac{1}{t}\sum_{t^{\prime}=1}^{t}R_{t^{\prime}},
(2)
which measures the average test risk of the model across the first
t
test time steps. If, at some time point
t^{*}
, the running risk
\bar{R}_{t^{*}}
exceeds the baseline source risk
R_{0}
by a pre-defined tolerance parameter
\epsilon_{\textrm{tol}}>0
, i.e., if
\bar{R}_{t^{*}}>R_{0}+\epsilon_{\textrm{tol}},
(3)
the risk monitor should ideally raise an alarm, indicating that deployment may be stopped and retraining or fine-tuning may be initiated.
Accordingly,
risk monitoring
addresses the binary hypothesis test defined by the hypotheses
\displaystyle\mathcal{H}_{0}:
\displaystyle\ \bar{R}_{t}\leq R_{0}+\epsilon_{\textrm{tol}},\quad\forall\,t\geq 1,
(4a)
\displaystyle\mathcal{H}_{1}:
\displaystyle\ \exists\,t^{*}\geq 1\text{ such that }\bar{R}_{t^{*}}>R_{0}+\epsilon_{\textrm{tol}}.
(4b)
The null hypothesis
\mathcal{H}_{0}
stipulates that the running test risk never exceeds the nominal risk
R_{0}
by more than the tolerance
\epsilon_{\textrm{tol}}
, while the alternative hypothesis
\mathcal{H}_{1}
covers cases in which the running risk exhibits a
harmful degradation
, i.e.,
\bar{R}_{t^{*}}>R_{0}+\epsilon_{\textrm{tol}}
for at least one time
t^{*}
.
In practice, the nominal risks
R_{t}
, for all indices
t=0,1,...
, are not directly accessible, but they can be estimated based on the available data via the
empirical risks
\hat{R}_{t}=\frac{1}{n_{t}}\sum_{i=1}^{n_{t}}\ell(f({x}_{t,i}),y_{t,i})
(5)
for
t=0,1,...
Using these estimates, at each time step
t
, a sequential test processes the first
t
estimates
\hat{R}_{0}
,
\hat{R}_{1}
,…,
\hat{R}_{t}
to output a decision
\Phi_{t}=\begin{cases}0,&\text{continue testing},\\
1,&\text{reject hypothesis $\mathcal{H}_{0}$}\quad\text{(alarm)}.\end{cases}
(6)
Accordingly, a decision
\Phi_{t}=1
indicates that there is enough evidence to reject the null hypothesis
\mathcal{H}_{0}
that no harmful shift has occurred,
while the decision
\Phi_{t}=0
causes the test to continue operating.
The design goal for the decision rule is to control the type-I error, i.e., the probability of producing decision
\Phi_{t}=1
when the null hypothesis
\mathcal{H}_{0}
of no harmful shifts holds true.
The type-I error requirement can be formulated as the inequality
\mathbb{P}_{\mathcal{H}_{0}}\left(\exists t\geq 1:\Phi_{t}=1\right)\leq\delta,
(7)
where
\delta\in(0,1)
is a user-defined value, and
P_{\mathcal{H}_{0}}
denotes any probability sequence
P_{1},P_{2},...
satisfying the condition (4a).
Besides guaranteeing the type-I error (7), one is also interested in reducing the average time required for correctly detecting a harmful shift. Formally, since the earliest time an alarm is produced as
\min\{t\geq 1:\Phi_{t}=1\}
, the average time to alarm under some sequence of distributions belonging to the alternative
\mathcal{H}_{1}
is defined as the expected stopping time
T=\mathbb{E}_{\mathcal{H}_{1}}\left[\min\left\{t\geq 1:\Phi_{t}=1\right\}\right].
(8)
2.2
Supervised Risk Monitoring
Given the estimated risks
\hat{R}_{0},\hat{R}_{1},\ldots,\hat{R}_{t}
in (5), the SRM method introduced in
(Podkopaev and Ramdas, 2022)
constructs an upper bound
{U}_{0}
on the source risk
{R}_{0}
and a lower bound
{L}_{t}
on the running test risk
\bar{R}_{t}
as
{U}_{0}=\hat{R}_{0}+w_{0},\quad\text{and}\quad{L}_{t}=\hat{\bar{R}}_{t}-w_{t}\>,
(9)
where
w_{0}
and
w_{t}
are finite-sample correction terms, and
\hat{\bar{R}}_{t}=t^{-1}\sum_{t^{\prime}=1}^{t}\hat{R}_{t^{\prime}}
denotes the empirical estimates of running risk.
The correction term
w_{0}
is selected so as to ensure that the true source risk
R_{0}
is upper bounded by
{U}_{0}
with probability no smaller than
1-\delta_{\textrm{S}}
, i.e.,
\mathbb{P}\left(R_{0}\leq U_{0}\right)\geq 1-\delta_{\textrm{S}},
(10)
for a given probability
\delta_{\textrm{S}}\in(0,1)
.
For example, the constant
w_{0}
can be obtained using Hoeffding’s inequality as
w_{0}=\sqrt{\ln\left(1/\delta_{\mathrm{S}}\right)/2n_{0}}
, given that the loss is assumed to be bounded
(Hoeffding, 1963; Einbinder et al., 2025)
.
In contrast, the constants
w_{t}
are chosen such that the sequence
\{L_{t}\}_{t\geq 1}
forms an
anytime-valid
lower bound on the running test risk. Formally, this requirement is captured by the inequality
\mathbb{P}\!\left(\bar{R}_{t}\geq L_{t},\ \forall\,t\geq 1\right)\geq 1-\delta_{\textrm{T}},
(11)
for some confidence level
\delta_{\textrm{T}}\in(0,1)
. This bound can be obtained by leveraging the following lemma.
Lemma 2.1
(Theorem 4, Conjugate-mixture empirical Bernstein (CM-EB) adapted from
(Howard et al., 2021)
)
.
Consider a sequence of random variables
\{Z_{t}\}_{t\geq 1}
with
Z_{t}\in[0,1]
for all
t\geq 1
. Define the time-averaged mean
\mu_{t}=t^{-1}\sum_{t^{\prime}=1}^{t}\mathbb{E}[Z_{t^{\prime}}]
,
and the empirical mean
\hat{\mu}_{t}=t^{-1}\sum_{t^{\prime}=1}^{t}Z_{t^{\prime}}
.
Let
\{\hat{Z}_{t}\}_{t\geq 1}
be any predictable sequence with
\hat{Z}_{t}\in[0,1]
, in the sense that each variable
\hat{Z}_{t}
is a function of the past variables
Z_{1},\dots,Z_{t-1}
only.
Define the function
u(V_{t})\!=\!\sup\!\ \{\!S:\!\int_{0}^{1}\!q(\lambda)\exp(\lambda S\!-\!\psi_{E}(\lambda)V_{t})\,\mathrm{d}\lambda\!<\!{1}/{\delta_{\emph{T}}}\}
for any distribution
q(\lambda)
on parameter
\lambda
, where the supremum is taken over the scalar
S
,
V_{t}=\sum_{{t^{\prime}}=1}^{t}\big(Z_{t^{\prime}}-\hat{Z}_{t^{\prime}}\big)^{2}
is the cumulative prediction error,
\psi_{E}(\lambda)\!=\!-\log(1\!-\!\lambda)\!-\!\lambda
, and
\delta_{\mathrm{T}}\in(0,1)
is a probability.
Then, we have the inequality
\mathbb{P}\!\left(\forall t\geq 1:\;\left|\mu_{t}-\hat{\mu}_{t}\right|<\frac{u(V_{t})}{t}\right)\geq 1-2\delta_{\mathrm{T}}.
(12)
In the context of SRM, the confidence sequence (12) in
Lemma
2.1
can be leveraged to obtain an anytime-valid lower confidence bound (9) with the correction term
w_{t}={u(V_{t})}/{t}
, where
V_{t}
becomes
V_{t}\!=\!\sum_{t^{\prime}=1}^{t}\big(\hat{R}_{t^{\prime}}\!-\!\hat{\bar{R}}_{t^{\prime}-1}\big)^{2}
.
By combining both bounds, i.e., (10) and (11), SRM obtains the decision variable as
\Phi_{t}=\mathbbm{1}\left[L_{t}>U_{0}+\epsilon_{\textrm{tol}}\right],
(13)
where
\mathbbm{1}[\cdot]
is the indicator function. As shown in Figure
, this decision rule triggers an alarm when the lower bound on the running test risk
\bar{R}_{t}
exceeds the upper bound on source risk
R_{0}
plus the tolerance margin
\epsilon_{\textrm{tol}}
. Accordingly, the average earliest detection time (8) is given by
T=\mathbb{E}\left[\min\left\{t\geq 1:L_{t}>U_{0}+\epsilon_{\textrm{tol}}\right\}\right].
(14)
SRM controls the type-I error (7) as long as one chooses the probabilities in (10) and (11) as
\delta=\delta_{\textrm{S}}+\delta_{\textrm{T}}
, as formalized in the following lemma.
Lemma 2.2
(Type-I Error Guarantee of SRM
(Podkopaev and Ramdas, 2022)
)
.
Given user-defined miscoverage levels
\delta_{\emph{S}},\delta_{\emph{T}}\in(0,1)
, satisfying the condition
\delta_{\emph{S}}+\delta_{\emph{T}}\in(0,1)
, SRM provides the following guarantee on the type-I error:
\mathbb{P}_{\mathcal{H}_{0}}\left(\exists t\geq 1,\Phi_{t}=1\right)\leq\delta_{\emph{S}}+\delta_{\emph{T}}.
(15)
3
Semi-Supervised Risk Monitoring
In many real-world deployments, acquiring labeled feedback is costly and time-consuming, while unlabeled test data are abundant and arrive continuously. As shown in Figure
, this motivates a semi-supervised paradigm where the risk monitor can leverage a small set of labeled samples, together with a larger pool of unlabeled observations.
To address this setting, in this section, we propose a generalization of SRM
(Podkopaev and Ramdas, 2022)
referred to as
prediction-powered semi-supervised risk monitoring
(PPRM), which can leverage both labeled and unlabeled data.
3.1
Prediction-Powered Risk Estimates
As shown in Figure
, at each time step
t=1,2,\dots
, the monitor observes a labeled batch
\mathcal{D}_{t}=\{({x}_{t,i},y_{t,i})\}_{i=1}^{n_{t}}
, as for SRM, along with an unlabeled batch
\tilde{\mathcal{D}}_{t}=\{\tilde{x}_{t,i}\}_{i=n_{t}+1}^{n_{t}+N_{t}}
of
N_{t}
inputs with no associated labels.
The main challenge in this setting lies in how to efficiently combine the labeled and unlabeled streams while controlling the type-I error (7). We emphasize that, unlike
(Schirmer et al., 2025; Amoukou et al., 2024)
, we assume that the number of labeled data points,
n_{t}
, is strictly positive. As we will see, this allows us to avoid the technical assumptions required in
(Schirmer et al., 2025)
about the error rates of the models used to generate synthetic labels for the unlabeled data.
PPRM incorporates PPI-based estimates of the risks
(Angelopoulos et al., 2023b)
into the SRM framework. Through PPI, PPRM constructs unbiased empirical estimates of the risks by: (i) using a predictive model
f_{\text{p}}(\cdot)
to generate synthetic labels for the unlabeled portion of the data; and then (ii) correcting for the systematic bias caused by the synthetic labels via calibration using the labeled samples. The labeling model
f_{\text{p}}(\cdot)
may be, for instance, a general-purpose expert, such as an LLM, or possibly the deployed model
f(\cdot)
itself. As anticipated, unlike
(Schirmer et al., 2025)
, we do not make any assumptions on the quality of the model
f_{\textrm{p}}(\cdot)
.
Write as
\tilde{y}_{t,i}
the synthetic labels for the unlabeled input
\tilde{x}_{t,i}
, i.e.,
\tilde{y}_{t,i}=\arg\max_{c}f_{\textrm{p}}\left({\tilde{x}}_{t,i}\right)[c],
(16)
where
f_{\textrm{p}}(\cdot)[c]
denotes the predicted probability assigned to class
c
by the predictive model
f_{\textrm{p}}(\cdot)
. Using PPI, the source risk
R_{0}
is estimated as
\displaystyle\hat{R}^{\textrm{PP}}_{0}
\displaystyle=\underbrace{\dfrac{\eta_{0}}{N_{0}}\sum_{j=n_{0}+1}^{n_{0}+N_{0}}\ell\big(f({\tilde{x}}_{0,j}),\tilde{y}_{0,j}\big)}_{\hat{R}^{\text{U}}_{0}}
(17)
\displaystyle+\underbrace{\frac{1}{n_{0}}\sum_{i=1}^{n_{0}}\ell\big(f({x}_{0,i}),y_{0,i}\big)-\frac{\eta_{0}}{n_{0}}\sum_{i=1}^{n_{0}}\ell\big(f({x}_{0,i}),\tilde{y}_{0,i}\big)}_{\hat{R}^{\textrm{rect}}_{0}},
where
\eta_{0}\geq 0
is a hyperparameter that controls the reliance of the estimate on the unlabeled data
(Angelopoulos et al., 2023a)
.
The first term
\hat{R}^{\textrm{U}}_{0}
in (17) is the empirical risk evaluated only with the unlabeled synthetic data, while the second term
\hat{R}^{\textrm{rect}}_{0}
estimates the bias introduced by the imputation. As proved in
(Angelopoulos et al., 2023b)
, using PPI++, the empirical estimate
\hat{R}^{\textrm{PP}}_{0}
is unbiased, in the sense that
\mathbb{E}[\hat{R}^{\textrm{PP}}_{0}]=R_{0}
, for any choice of hyperparameter
\eta_{0}\geq 0
.
In a similar way, the running test risk
\bar{R}_{t}
is estimated as
\displaystyle\hat{\bar{R}}_{t}^{\textrm{PP}}=\;\frac{1}{t}\sum_{{t^{\prime}}=1}^{t}\Bigg[\underbrace{\frac{\eta_{t^{\prime}}}{N_{t^{\prime}}}\sum_{j=n_{t^{\prime}}+1}^{n_{t^{\prime}}+N_{t^{\prime}}}\ell(f(\tilde{x}_{{t^{\prime}},j}),\tilde{y}_{{t^{\prime}},j})}_{\hat{R}^{\textrm{U}}_{t^{\prime}}}
(18)
\displaystyle+\underbrace{\frac{1}{n_{t^{\prime}}}\sum_{i=1}^{n_{t^{\prime}}}\ell(f(x_{{t^{\prime}},i}),y_{{t^{\prime}},i})-\frac{\eta_{t^{\prime}}}{n_{t^{\prime}}}\sum_{i=1}^{n_{t^{\prime}}}\ell(f(x_{{t^{\prime}},i}),\tilde{y}_{{t^{\prime}},i})}_{\hat{R}^{\textrm{rect}}_{t^{\prime}}}\Bigg],
using both labeled and unlabeled data, where
\eta_{t}
plays the same role as in (17). Furthermore, as in (17), the first term
\hat{R}^{\textrm{U}}_{t}
in (18) denotes the empirical risk evaluated only with the unlabeled synthetic data at time step
t
, while the second term
\hat{R}^{\textrm{rect}}_{t}
estimates the bias introduced by the imputation. We write
\hat{R}^{\textrm{PP}}_{t}=\hat{R}^{\textrm{U}}_{t}+\hat{R}^{\textrm{rect}}_{t}
(19)
for the contribution to the estimate (18) corresponding to each time step
t
.
Lemma 3.1
(Unbiasedness of PPRM)
.
Assume that the hyperparameters
\{\eta_{t}\}_{t\geq 1}
form a predictable sequence, in the sense that each hyperparameter
\eta_{t}
is a function only of the data
\{\mathcal{D}_{t^{\prime}},\tilde{\mathcal{D}}_{t^{\prime}},\{\tilde{y}_{t^{\prime},i}\}_{i=1}^{n_{t^{\prime}}\!+\!N_{t^{\prime}}}\}_{t^{\prime}\!=\!1}^{t\!-\!1}
observed strictly before time
t
. Then, the quantity (18) is an unbiased estimate of the corresponding true risk, i.e.,
\mathbb{E}\big[\hat{\bar{R}}^{\emph{PP}}_{t}\big]=\bar{R}_{t}.
(20)
The proof can be found in Appendix
A.1
.
3.2
Upper and Lower PPRM Bounds
As in SRM, PPRM obtains an upper bound
U_{0}^{\textrm{PP}}
on the source risk, as well as a lower confidence bound
L_{t}^{\textrm{PP}}
for
t=1,2,\ldots
, which are denoted respectively by
{U}_{0}^{\textrm{PP}}=\hat{R}_{0}^{\textrm{PP}}+w_{0}^{\textrm{PP}}\quad\text{and}\quad{L}_{t}^{\textrm{PP}}=\hat{\bar{R}}_{t}^{\textrm{PP}}-w_{t}^{\textrm{PP}}\>.
(21)
The correction term
w_{0}^{\textrm{PP}}
is designed to ensure the inequality
\mathbb{P}(R_{0}\!\leq\!U_{0}^{\textrm{PP}})\!\geq\!1\!-\!\delta_{\textrm{S}}
,
and we adopt here the betting-based method from
(Einbinder et al., 2025)
. Further details are provided in Appendix
A.2
.
To compute the correction terms
w_{t}^{\textrm{PP}}
, we apply the confidence sequence bound in Lemma
2.1
to the PPI-based risk estimates.
To this end, since the prediction-powered estimates
\hat{R}_{t}^{\textrm{PP}}
take values in the interval
[-\eta_{t},\eta_{t}+1]
, we first apply an affine normalization
g_{a}(\cdot)
, i.e.,
g_{a}(\ell)\!=\!(\ell\!+\!\eta_{\textrm{max}})/(1\!+\!2\eta_{\textrm{max}})
, to the loss function
\ell
so that each estimate
\hat{R}_{t}^{\textrm{PP}}
lies in the range
[0,1]
, where
\eta_{\textrm{max}}
is the maximum permitted value for
\eta_{t}
.
With the rescaled loss, the correction term
w_{t}^{\textrm{PP}}
is given by
w_{t}^{\textrm{PP}}\!=\!u(V_{t}^{\textrm{PP}})/t
, with
u(V_{t}^{\textrm{PP}})\!=\!\sup\{S\!:\!\int_{0}^{1}q(\lambda)\exp(\lambda S\!-\!\psi_{E}(\lambda)V_{t}^{\textrm{PP}})\,\mathrm{d}\lambda\!<\!1/\delta_{\textrm{T}}\}
, where
V_{t}^{\textrm{PP}}\!=\!\sum_{t^{\prime}=1}^{t}\bigl(\hat{R}_{t^{\prime}}^{\textrm{PP}}\!-\!\hat{\bar{R}}_{t^{\prime}-1}^{\textrm{PP}}\bigr)^{2}.
(22)
The term
V_{t}^{\textrm{PP}}
in (22) is the cumulative squared prediction error for the estimate
\hat{R}_{t^{\prime}}^{\textrm{PP}}
in (19) when using the corresponding time-average up to time
t^{\prime}\!-\!1
, i.e.,
\hat{\bar{R}}_{t^{\prime}-1}^{\textrm{PP}}
in (18), as the prediction.
PPRM then applies the decision rule as
\Phi_{t}^{\textrm{PP}}=\mathbbm{1}\left[L_{t}^{\textrm{PP}}>U_{0}^{\textrm{PP}}+\epsilon_{\textrm{tol}}\right].
(23)
Following the same argument as Lemma
2.2
, we have the following result.
Theorem 3.2
(Type-I Error Guarantee of PPRM)
.
Given user-defined miscoverage levels
\delta_{\emph{S}},\delta_{\emph{T}}\in(0,1)
, satisfying the condition
\delta_{\emph{S}}+\delta_{\emph{T}}\in(0,1)
, let
\{\eta_{t}\}_{t\geq 1}
be any predictable sequence, as defined in Lemma
3.1
. Then, PPRM provides the following guarantee
on the type-I error:
\mathbb{P}_{\mathcal{H}_{0}}\left(\exists\,t\geq 1:\,\Phi_{t}^{\emph{PP}}=1\right)\leq\delta_{\emph{S}}+\delta_{\emph{T}}.
(24)
To prove this result, it suffices to note that, as long as the sequence
\{\eta_{t}\}_{t\geq 1}
is
predictable, the prediction-powered risk estimates (17) and (18) are unbiased by Lemma
3.1
.
The proof then follows directly from
Lemma
2.2
.
3.3
Optimizing the Reliance on Unlabeled Data
The prediction-powered risk estimate
\hat{R}_{t}^{\textrm{PP}}
relies on the unlabeled data to an extent that depends on the hyperparameter
\eta_{t}
by leveraging the result in Lemma
3.1
, which allows
\eta_{t}
to be a function of past data up to time
t\!-\!1
.
To this end, we propose to minimize the average value of the prediction error
V_{t}^{\textrm{PP}}
in (22), which determines the tightness of the lower bound
L_{t}^{\textrm{PP}}
in (21). A tighter bound ensures that, when the alternative hypothesis (4b) holds, the PPRM test (23) signals a harmful degradation, setting
\Phi_{t}^{\textrm{PP}}\!=\!1
, at an earlier time
t
.
Lemma 3.3
(Optimal Choice of \eta_{t} for Variance Reduction)
.
Let
\hat{R}_{t}^{\emph{PP}}(\eta_{t})
denote the prediction-powered empirical risk (19) at time
t
, which is parameterized by the hyperparameter
\eta_{t}
.
The value of hyperparameter
\eta_{t}
that minimizes the variance, i.e.,
\eta_{t}^{*}=\arg\min_{\eta_{t}\geq 0}\mathbb{E}\!\left[\bigl(\hat{R}_{t}^{\emph{PP}}(\eta_{t})-\hat{\bar{R}}_{t-1}^{\emph{PP}}\bigr)^{2}\right],
(25)
is given by
\eta_{t}^{*}=\frac{\operatorname{Cov}(u_{t},\tilde{u}_{t}^{\mathrm{L}})}{(1+\frac{n_{t}}{N_{t}})\operatorname{Var}(\tilde{u}_{t}^{\mathrm{U}})},
(26)
where
u_{t}=\ell(f(x_{t}),y_{t})
denotes the true risk on a labeled sample;
\tilde{u}_{t}^{\mathrm{L}}=\ell(f(x_{t}),f_{\emph{p}}(x_{t}))
and
\tilde{u}_{t}^{\mathrm{U}}=\ell(f(\tilde{x}_{t}),f_{\emph{p}}(\tilde{x}_{t}))
denote the surrogate risk on a labeled sample and on an unlabeled sample, respectively; and
\operatorname{Var}(\cdot)
and
\operatorname{Cov}(\cdot)
represent variance and covariance.
The proof of this result, which follows in a manner similar to
(Angelopoulos et al., 2023a)
, can be found in Appendix
A.3
.
The hyperparameter
\eta_{t}^{*}
in (26) depends on the true data distribution, which is unknown. Thus, we propose to estimate it using the data up to time step
t\!-\!1
based on a sliding-window approach with fixed window size
L
. This produces a predictable sequence
\{\eta_{t}\}_{t\geq 1}
, satisfying the assumption of Theorem
3.2
. Specifically, we consider the following plug-in estimator
\eta_{t}=\frac{\widehat{\operatorname{Cov}}_{\mathcal{H}_{t-1}}(u,\tilde{u})}{\left(1+\frac{|\mathcal{H}_{t-1}|}{|\tilde{\mathcal{H}}_{t-1}|}\right)\widehat{\operatorname{Var}}_{\tilde{\mathcal{H}}_{t-1}}\left(\tilde{u}\right)},
(27)
where
\widehat{\operatorname{Cov}}_{\mathcal{H}_{t-1}}(u,\tilde{u})
and
\widehat{\operatorname{Var}}_{\tilde{\mathcal{H}}_{t-1}}\left(\tilde{u}\right)
denote the empirical estimates using the data
\mathcal{H}_{t-1}=\bigcup_{t^{\prime}=t-L}^{t-1}\mathcal{D}_{t^{\prime}},\quad\text{and}\quad\tilde{\mathcal{H}}_{t-1}=\bigcup_{t^{\prime}=t-L}^{t-1}\tilde{\mathcal{D}}_{t^{\prime}},
(28)
with
|\mathcal{H}_{t-1}|
and
|\tilde{\mathcal{H}}_{t-1}|
denoting the total numbers of historical labeled and unlabeled samples available before time
t
, respectively.
We further provide a theoretical analysis of the detection delay, under a simplified stationary regime with binary risks in Appendix
A.4
.
4
Related Works
Sequential Testing-Based Risk Monitoring:
SRM was introduced by
(Podkopaev and Ramdas, 2022)
to tackle the problem of detecting harmful distribution shifts in a sequential setting. SRM builds confidence bounds for model risk using labeled calibration data
(Podkopaev and Ramdas, 2022)
and incoming test data, and provides time-uniform guarantees on type-I error control. Subsequently,
Amoukou et al.
extended this line of work to the unlabeled production-data setting, while
Schirmer et al.
further explored scenarios involving test-time adaptation models without access to labels. Related approaches by
(Bar et al., 2024)
and
(Vovk et al., 2021)
focus on identifying changes in certain statistical properties. However, such changes do not necessarily indicate harmful deviations.
Beyond tracking running risk, several studies have examined the instantaneous true risk with similar type-I error guarantees
(Xu et al., 2024; Zecchin et al., 2025; Timans et al., 2025; Shekhar and Ramdas, 2023; Vovk and Wang, 2021)
. These methods are built around the testing-by-betting framework
(Ramdas and Wang, 2025)
. However, we target persistent performance degradation, while these methods are designed to detect instantaneous violations.
Figure 2:
Risk estimates as a function of time
t
and average time to alarm for an image classification task under increasing shift severity:
(a) binary loss monitored with an external predictor;
(b) squared loss monitored using labels produced by the deployed model itself.
Prediction-Powered Inference:
In many practical settings, models can access large pools of unlabeled data but only a limited amount of costly ground-truth labels. PPI was introduced to make use of predictions from an auxiliary model to facilitate statistical inference. Its key idea is to construct prediction-powered estimators that remain unbiased for the target risk, while using predictive information to reduce variance. The framework has since been extended in several directions, including robust PPI++
(Angelopoulos et al., 2023a)
, local PPI
(Gu and Xia, 2024)
, anytime-valid PPI
(Kilian et al., 2025)
, active PPI
(Zrnic and Candès, 2024)
, and FAB-PPI
(Cortinovis and Caron, 2025)
. More recently,
Csillag et al.
introduced prediction-powered e-values
(Shafer and Vovk, 2019)
for sequential change-point detection
(Shekhar and Ramdas, 2023; Shin et al., 2022)
. Their approach also builds on the testing-by-betting paradigm
(Ramdas and Wang, 2025)
, and therefore differs from our proposed PPRM both in the hypothesis being tested and in the experimental setup.
LLM Performance Evaluation:
Selecting an appropriate model from a number of candidates requires reliable estimates of each model’s performance. Traditional evaluation methods depend on deploying models and gathering empirical evidence through real-world testing, which is often expensive to deploy. To mitigate this burden, recent studies have investigated autoevaluation techniques that automate assessment using synthetic data or automated tools, thereby reducing the need for human involvement
(Chaganty et al., 2018; Zheng et al., 2023)
.
PPI has also been applied to improve the efficiency of model evaluation
(Fisch et al., 2024; Park et al., 2025; Einbinder et al., 2025)
. Nonetheless, the problem of monitoring a deployed model using PPI has not been explored.
5
Experiments
In this section, we present two case studies, including image classification and LLM monitoring, and defer additional results including for a telecommunications task, to the appendix.
Figure 3:
Performance for the LLM QA task under prompt shifts introduced at time
t=200
.
Baselines
We compare PPRM to several baseline monitors:
•
Ideal PPRM
: This is an idealized case where PPRM has full access to the true labels of both labeled dataset
\mathcal{D}_{t}
and unlabeled dataset
\tilde{\mathcal{D}}_{t}
. This produces the performance upper bound corresponding to full label access.
•
SRM
(Podkopaev and Ramdas, 2022)
: As discussed in Section
2.2
, SRM uses only the labeled data
\mathcal{D}_{t}
.
•
Unsupervised risk monitoring (URM)
(Schirmer et al., 2025)
: This scheme uses only unlabeled data and is reviewed in Appendix
B
.
We focus on binary
0
–
1
loss and squared loss.
Unless otherwise specified, we set
\delta=\delta_{\textrm{S}}+\delta_{\textrm{T}}=0.25
, allocating most of the budget to controlling the test risk, i.e.,
\delta_{\textrm{T}}=0.2
and
\delta_{\textrm{S}}=0.05
. We set
\eta_{0}=1
in (17), and the window length
L
to obtain
\eta_{t}
in (27) is set to
60
, numerical validations of these choices can be found in Appendix
D.1
. The numbers of labeled data samples and unlabeled data samples at each time step are set to
1
and
15
. To compute all the lower bounds of the running risk, we adopt the CM-EB method reviewed in Lemma
2.1
. For all figures, we show confidence intervals constructed spanning one standard deviation of the empirical distribution across all runs on both sides of the mean.
5.1
Sequential Risk Monitoring on Image Data
5.1.1
Problem Formulation
We first consider the task of risk monitoring in the vision domain, replicating the setting of
(Schirmer et al., 2025)
, where the CIFAR10-C
(Hendrycks and Dietterich, 2019)
dataset was considered. Accordingly, the test-time distribution is obtained by applying Gaussian noise with three levels of severity to the image distribution corresponding to the nominal condition. We monitor the performance of a ResNet-32 model
(He et al., 2016)
trained on nominal data. To generate synthetic labels for unlabeled samples, we employ the ResNet-1201 model trained on the same dataset.
Figure 4:
Average time to alarm for the LLM QA task using different predictors
f_{\textrm{p}}(\cdot)
to produce synthetic labels.
5.1.2
Results
Monitoring with Drifting Data Distribution:
To simulate an increasing test risk, we gradually increase the severity level of Gaussian noise corruption from nominal data, i.e., severity level
0
, up to severity level
2
.
Figure 2 (a) shows the results of the estimated test risk and of the alarm decisions. As the corruption intensifies, the true risk exceeds the safe threshold at time
t\!=\!600
, PPRM raises an alarm at around time
t\!=\!760
, whereas SRM reacts more slowly, at around time
t\!=\!800
due to high estimation variance, and URM fails to trigger an alarm within the considered time period.
Monitoring with Self-Synthetic Strategy:
The proposed PPRM can also be applied when the model itself is used to provide label surrogates for the unlabeled data. For this setting, focusing on
C
-class classification,
we consider the squared loss between labels and model confidence levels, i.e.,
\ell(f(x),y)\!=\!\frac{1}{2}\sum_{c=1}^{C}(f(x)_{c}\!-\!\mathbbm{1}[y\!=\!c])^{2}
, as the loss function. This squared loss corresponds to the Brier score when averaged across samples, and it was also adopted in
(Schirmer et al., 2025)
.
As seen in Figure 2 (b), even without an additional predictor, PPRM can reduce the time to alarm by making use of the model’s intrinsic confidence signals. In particular, an alarm is raised at around time
t\!=\!760
, while SRM requires
t=790
time steps.
5.2
Monitoring an LLM with Limited Human-Labeled Data
5.2.1
Problem Formulation
We now focus on an LLM deployed to address
question-answering
(QA) tasks. We draw from several established benchmarks, including MMLU
(Hendrycks et al., 2021)
, CMExam
(Liu et al., 2023)
, CommonsenseQA
(Talmor et al., 2019)
, Social-IQA, and PubMedQA
(Jin et al., 2019)
.
We use Qwen2-VL-2B
(Wang et al., 2024)
as the deployed model whose performance we aim to monitor. For the monitoring LLM, we experiment with a diverse set of both open-source and closed-source models.
5.2.2
Results
To simulate an environment in which risk increases over time, starting from time
t\!=\!200
, we introduce more complex prompts that the deployed model cannot answer correctly, causing the model’s performance to deteriorate. These complex prompts are obtained by retaining cases where a different model, namely Qwen2.5-VL-7B, fails to produce correct answers.
Monitoring with Drifting Data Distribution:
As shown in Figure 3 , confirming the results in the previous example, PPRM significantly reduces the average time to alarm. In particular, while the true risk exceeds the allowed level at
t\!=\!280
, PPRM raises an alarm at around time
t\!=\!500
, whereas SRM raises an alarm at around time
t\!=\!530
.
We then further assess how the accuracy of the LLM predictor affects performance by testing a range of different labeling models, namely Qwen2.5-VL-3B (abbreviated as “Q2.5-VL3B”), Qwen2.5-VL-32B (abbreviated as “Q2.5-VL32B”)
(Bai et al., 2025)
, Qwen2-VL-7B (abbreviated as “Q2-VL7B”)
(Wang et al., 2024)
, GPT4.1-nano (abbreviated as “GPT4.1-n”), and GPT4.1
(Achiam et al., 2023)
.
The results are shown in Figure 4 by ordering the models from least to most powerful. As seen in the figure,
when the LLM predictor exhibits higher accuracy, i.e., for larger models, the estimated accuracy of synthetic labels increases, leading to a tighter bound on the monitored model’s running risk, and consequently to a lower average time to alarm.
In Figure 5 , we report the false-alarm rate as a function of time
t
for the LLM QA task. We run multiple independent trials under the null setting, and record the entire trajectory of the lower confidence bound over time. For each trial, we check at every time step whether the bound has crossed the risk threshold, and convert this trajectory into a cumulative indicator, which equals one once a crossing has occurred at any previous time. The reported curve is obtained by averaging these cumulative indicators across trials at each time step. The results illustrate that the false-alarm rate remains below the target level across all time steps, thereby validating the theoretical guarantees of PPRM.
Figure 5:
False-alarm rate as a function of time
t
for the LLM QA task. The dashed line indicates the target false-alarm rate
\delta_{\mathrm{T}}=0.15
.
Table 1:
Average time to alarm for PPRM with a fixed or an adaptive hyperparameter
\eta_{t}
.
Predictor
f_{\textrm{p}}(\cdot)
PPRM (Fixed)
PPRM (Adaptive)
Q2.5-VL3B
527.80
508.83
Q2-VL7B
519.08
498.31
GPT4.1-n
491.80
476.98
Q2.5-VL32B
486.22
471.85
GPT4.1
460.69
449.88
Effect of Hyperparameter Optimization:
Table 1 reports the average time to alarm for PPRM with a fixed hyperparameter
\eta_{t}=1
and with an adaptive hyperparameter
\eta_{t}
in (27). The two PPRM variants are identified as “PPRM (Fixed)” and “PPRM (Adaptive)”, respectively. We consider the different imputation predictors
f_{\textrm{p}}(\cdot)
considered in Figure 4 . The table illustrates the benefits of the adaptive selection, particularly for small- and medium-sized predictors. For example, for the Qwen2-VL-7B model, the average time to alarm decreases from around
519
to around
498
thanks to adaptive hyperparameter selection. It is also observed that, as the predictor becomes stronger, the performance gap between PPRM with fixed and adaptive hyperparameters gradually narrows. This demonstrates that adaptive hyperparameter selection, determining the reliance on unlabeled data, is particularly important for less effective imputation models.
6
Conclusion
In this work, we have presented a novel framework for sequential semi-supervised risk monitoring, extending classical supervised approaches to leverage both labeled and unlabeled data. By integrating PPI, our method, termed PPRM, provided unbiased estimates of the running test risk, while maintaining rigorous statistical guarantees on the type-I error. Through extensive experiments on three tasks, we have demonstrated that the proposed approach consistently achieved timely detection of harmful distribution shifts and outperformed purely supervised or unsupervised baselines. Future work may consider methods that integrate PPRM with unsupervised techniques such as test-time adaptation. Other interesting research directions include applications to engineering problems, e.g., in robotics.
Acknowledgements
The work of Guangyi Zhang and Yunlong Cai was supported in part by the National Natural Science Foundation of China under Grant 62571477, and in part by Zhejiang Provincial Key Laboratory of Multi-Modal Communication Networks and Intelligent Information Processing, Hangzhou 310027, China. The work of Osvaldo Simeone was supported by the European Research Council (ERC) under the European Union’s Horizon Europe Programme (grant agreement No. 101198347), by an Open Fellowship of the EPSRC (EP/W024101/1), and by the EPSRC project (EP/X011852/1).
Impact Statement
This paper presents work whose goal is to advance the field
of machine learning. The impact of this paper may be more pronounced in engineering applications, such as telecommunication and robotics.
References
J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat,
et al.
GPT-4 technical report
.
arXiv preprint arXiv:2303.08774
.
Cited by:
§5.2.2
.
S. I. Amoukou, T. Bewley, S. Mishra, F. Lecue, D. Magazzeni, and M. Veloso
Sequential harmful shift detection without labels
.
In
Advances in Neural Information Processing Systems
,
Vol.
37
,
pp. 129279–129302
.
Cited by:
§1
,
§1
,
§3.1
,
§4
.
A. N. Angelopoulos, J. C. Duchi, and T. Zrnic
PPI++: efficient prediction-powered inference
.
arXiv preprint arXiv:2311.01453
.
Cited by:
§A.3
,
§3.1
,
§3.3
,
§4
.
A. N. Angelopoulos, S. Bates, C. Fannjiang, M. I. Jordan, and T. Zrnic
Prediction-powered inference
.
Science
382
(6671),
pp. 669–674
.
Cited by:
§1
,
§3.1
,
§3.1
.
S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang,
et al.
Qwen2.5-VL technical report
.
arXiv preprint arXiv:2502.13923
.
Cited by:
§5.2.2
.
Y. Bar, S. Shaer, and Y. Romano
Protected test-time adaptation via online entropy matching: a betting approach
.
In
Advances in Neural Information Processing Systems
,
Vol.
37
,
pp. 85467–85499
.
Cited by:
§4
.
A. Chaganty, S. Mussmann, and P. Liang
The price of debiasing automatic metrics in natural language evaluation
.
In
Proceedings of the Annual Meeting of the Association for Computational Linguistics
,
pp. 643–653
.
Cited by:
§4
.
S. Cortinovis and F. Caron
FAB-PPI: frequentist, assisted by bayes, prediction-powered inference
.
In
Proceedings of the International Conference on Machine Learning
,
Vol.
267
,
pp. 11328–11356
.
Cited by:
§4
.
D. Csillag, C. J. Struchiner, and G. T. Goedert
Prediction-powered e-values
.
arXiv preprint arXiv:2502.04294
.
Cited by:
§4
.
F. A. Dietrich, T. Ivanov, and W. Utschick
Estimation of channel and noise correlations for MIMO channel estimation
.
In
Proceedings of the International ITG Workshop on Smart Antennas
,
Ulm, Germany
.
Cited by:
§C.3.1
.
B. Einbinder, L. Ringel, and Y. Romano
Semi-supervised risk control via prediction-powered inference
.
IEEE Transactions on Pattern Analysis and Machine Intelligence
47
(12),
pp. 11080–11090
.
Cited by:
§2.2
,
§3.2
,
§4
.
A. Fisch, J. Maynez, R. A. Hofer, B. Dhingra, A. Globerson, and W. W. Cohen
Stratified prediction-powered inference for effective hybrid evaluation of language models
.
In
Advances in Neural Information Processing Systems
,
Vol.
37
,
pp. 111489–111514
.
Cited by:
§4
.
Y. Gu and D. Xia
Local prediction-powered inference
.
arXiv preprint arXiv:2409.18321
.
Cited by:
§4
.
H. Guan, D. Bates, and L. Zhou
Keeping medical AI healthy and trustworthy: a review of detection and correction methods for system degradation
.
IEEE Transactions on Biomedical Engineering
(
),
pp. 1–16
.
Cited by:
§1
.
K. He, X. Zhang, S. Ren, and J. Sun
Deep residual learning for image recognition
.
In
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
,
pp. 770–778
.
Cited by:
§5.1.1
.
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt
Measuring massive multitask language understanding
.
In
International Conference on Learning Representations
,
Cited by:
§5.2.1
.
D. Hendrycks and T. Dietterich
Benchmarking neural network robustness to common corruptions and perturbations
.
In
International Conference on Learning Representations
,
Cited by:
§5.1.1
.
W. Hoeffding
Probability inequalities for sums of bounded random variables
.
Journal of the American statistical association
58
(301),
pp. 13–30
.
Cited by:
§2.2
.
S. R. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon
Time-uniform Chernoff bounds via nonnegative supermartingales
.
Probability Surveys
17
(none),
pp. 257–317
.
Cited by:
§A.4
,
Proposition A.1
.
S. R. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon
Time-uniform, nonparametric, nonasymptotic confidence sequences
.
The Annals of Statistics
49
(2),
pp. 1055–1080
.
Cited by:
Lemma 2.1
.
Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu
Pubmedqa: a dataset for biomedical research question answering
.
In
Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)
,
pp. 2567–2577
.
Cited by:
§5.2.1
.
V. Kilian, S. Cortinovis, and F. Caron
Anytime-valid, bayes-assisted, prediction-powered inference
.
In
Advances in Neural Information Processing Systems
,
D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Eds.)
,
Vol.
38
,
pp. 900–918
.
Cited by:
§4
.
Z. Lipton, Y. Wang, and A. Smola
Detecting and correcting for label shift with black box predictors
.
In
Proceedings of the International Conference on Machine Learning
,
pp. 3122–3130
.
Cited by:
§1
.
J. Liu, P. Zhou, Y. Hua, D. Chong, Z. Tian, A. Liu, H. Wang, C. You, Z. Guo, L. Zhu, and M. L. Li
Benchmarking large language models on CMExam - a comprehensive Chinese medical exam dataset
.
In
Advances in Neural Information Processing Systems
,
Vol.
36
,
pp. 52430–52452
.
Cited by:
§5.2.1
.
S. Park, M. Zecchin, and O. Simeone
Adaptive prediction-powered autoeval with reliability and efficiency guarantees
.
In
Advances in Neural Information Processing Systems
,
Vol.
38
,
pp. 158758–158784
.
Cited by:
§4
.
A. Podkopaev and A. Ramdas
Tracking the risk of a deployed model and detecting harmful distribution shifts
.
In
International Conference on Learning Representations
,
Cited by:
§1
,
§2.1
,
§2.2
,
Lemma 2.2
,
§2
,
§3
,
§4
,
2nd item
.
A. Ramdas and R. Wang
Hypothesis testing with e-values
.
Foundations and Trends® in Statistics
1
(1-2),
pp. 1–390
.
Cited by:
§4
,
§4
.
M. Schirmer, M. Jazbec, C. Andersson Naesseth, and E. Nalisnick
Monitoring risks in test-time adaptation
.
In
Advances in Neural Information Processing Systems
,
Vol.
38
,
pp. 80915–80948
.
Cited by:
Assumption B.1
,
Appendix B
,
Appendix B
,
Appendix B
,
§D.4
,
§D.4
,
§1
,
§3.1
,
§3.1
,
§4
,
3rd item
,
§5.1.1
,
§5.1.2
.
A. Sellergren, S. Kazemzadeh, T. Jaroensri, A. Kiraly, M. Traverse, T. Kohlberger, S. Xu, F. Jamil, C. Hughes, C. Lau,
et al.
Medgemma technical report
.
arXiv preprint arXiv:2507.05201
.
Cited by:
§C.2
.
G. Shafer and V. Vovk
Game‑theoretic foundations for probability and finance
.
Wiley Series in Probability and Statistics
,
John Wiley & Sons
,
Hoboken, NJ
.
Cited by:
§4
.
S. Shekhar and A. Ramdas
Sequential changepoint detection via backward confidence sequences
.
In
Proceedings of the International Conference on Machine Learning
,
Vol.
202
,
pp. 30908–30930
.
Cited by:
§4
,
§4
.
J. Shin, A. Ramdas, and A. Rinaldo
E-detectors: a nonparametric framework for sequential change detection
.
arXiv preprint arXiv:2203.03532
.
Cited by:
§4
.
O. Simeone, S. Park, and M. Zecchin
Conformal calibration: ensuring the reliability of black-box AI in wireless systems
.
IEEE Communications Magazine
(
),
pp. 1–9
.
Cited by:
§1
.
W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng
Sparsedrive: end-to-end autonomous driving via sparse scene representation
.
In
IEEE International Conference on Robotics and Automation
,
pp. 8795–8801
.
Cited by:
§1
.
A. Talmor, J. Herzig, N. Lourie, and J. Berant
Commonsenseqa: a question answering challenge targeting commonsense knowledge
.
In
Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)
,
pp. 4149–4158
.
Cited by:
§5.2.1
.
A. Timans, R. Verma, E. Nalisnick, and C. A. Naesseth
On continuous monitoring of risk violations under unknown shift
.
In
Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence
,
Cited by:
§4
.
V. Vovk, I. Petej, I. Nouretdinov, E. Ahlberg, L. Carlsson, and A. Gammerman
Retrain or not retrain: conformal test martingales for change-point detection
.
In
Conformal and Probabilistic Prediction and Applications
,
pp. 191–210
.
Cited by:
§4
.
V. Vovk and R. Wang
E-values: calibration, combination and applications
.
The Annals of Statistics
49
(3),
pp. 1736–1754
.
Cited by:
§4
.
P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan,
et al.
Qwen2-VL: enhancing vision-language model’s perception of the world at any resolution
.
arXiv preprint arXiv:2409.12191
.
Cited by:
§5.2.1
,
§5.2.2
.
I. Waudby-Smith and A. Ramdas
Estimating means of bounded random variables by betting
.
Journal of the Royal Statistical Society: Series B (Statistical Methodology)
86
(1),
pp. 1–27
.
Cited by:
§A.2
.
Z. Xu, N. Karampatziakis, and P. Mineiro
Active, anytime-valid risk controlling prediction sets
.
In
Advances in Neural Information Processing Systems
,
Vol.
37
,
pp. 60110–60132
.
Cited by:
§4
.
M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine
Robotic control via embodied chain-of-thought reasoning
.
In
Proceedings of the Conference on Robot Learning
,
Vol.
270
,
pp. 3157–3181
.
Cited by:
§1
.
S. Zeb, F. Nizamullah, N. Abbasi, and M. Fahad
AI in healthcare: revolutionizing diagnosis and therapy
.
International Journal of Multidisciplinary Sciences and Arts
3
(3),
pp. 118–128
.
Cited by:
§1
.
M. Zecchin, S. Park, O. Simeone, M. Kountouris, and D. Gesbert
Robust bayesian learning for reliable wireless ai: framework and applications
.
IEEE Transactions on Cognitive Communications and Networking
9
(4),
pp. 897–912
.
Cited by:
§1
.
M. Zecchin, S. Park, and O. Simeone
Adaptive learn-then-test: statistically valid and efficient hyperparameter selection
.
In
Proceedings of the International Conference on Machine Learning
,
Vol.
267
,
pp. 74018–74036
.
Cited by:
§4
.
L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing,
et al.
Judging LLM-as-a-judge with MT-bench and chatbot arena
.
In
Advances in neural information processing systems
,
Vol.
36
,
pp. 46595–46623
.
Cited by:
§1
,
§4
.
T. Zrnic and E. J. Candès
Active statistical inference
.
In
Proceedings of the International Conference on Machine Learning
,
Vol.
235
,
pp. 62993–63010
.
Cited by:
§4
.
Appendix A
Proofs and Analysis
A.1
Proof of Lemma
3.1
We aim to show the equality
\mathbb{E}[\hat{\bar{R}}_{t}^{\textrm{PP}}]=\bar{R}_{t}=t^{-1}\sum_{t^{{}^{\prime}}=1}^{t}R_{t^{{}^{\prime}}}
, where
R_{t^{{}^{\prime}}}=\mathbb{E}[\ell(f(x_{t^{\prime}}),y_{t^{\prime}})]
.
Recall that we have
\displaystyle\hat{\bar{R}}_{t}^{\textrm{PP}}
\displaystyle=\frac{1}{t}\sum_{t^{{}^{\prime}}=1}^{t}\left[\hat{R}_{t^{{}^{\prime}}}^{\textrm{U}}+\hat{R}_{t^{{}^{\prime}}}^{\textrm{rect}}\right]
(29)
\displaystyle=\frac{1}{t}\sum_{t^{{}^{\prime}}=1}^{t}\left[\frac{\eta_{t^{{}^{\prime}}}}{N_{t^{{}^{\prime}}}}\sum_{j=n_{t^{{}^{\prime}}}+1}^{n_{t^{{}^{\prime}}}+N_{t^{{}^{\prime}}}}\ell\left(f(\tilde{x}_{t^{{}^{\prime}},j}),\tilde{y}_{t^{{}^{\prime}},j}\right)+\frac{1}{n_{t^{{}^{\prime}}}}\sum_{i=1}^{n_{t^{{}^{\prime}}}}\ell\left(f(x_{t^{{}^{\prime}},i}),y_{t^{{}^{\prime}},i}\right)-\frac{\eta_{t^{{}^{\prime}}}}{n_{t^{{}^{\prime}}}}\sum_{i=1}^{n_{t^{{}^{\prime}}}}\ell\left(f(x_{t^{{}^{\prime}},i}),\tilde{y}_{t^{{}^{\prime}},i}\right)\right].
By linearity of expectation, we have
\mathbb{E}\left[\hat{\bar{R}}_{t}^{\textrm{PP}}\right]=\frac{1}{t}\sum_{t^{{}^{\prime}}=1}^{t}\mathbb{E}\left[\hat{R}_{t^{{}^{\prime}}}^{\textrm{U}}+\hat{R}_{t^{{}^{\prime}}}^{\textrm{rect}}\right].
(30)
For each term, we further have
\mathbb{E}[\hat{R}^{\text{U}}_{t}+\hat{R}^{\textrm{rect}}_{t}]=\mathbb{E}_{\eta_{t}}[\mathbb{E}[\hat{R}^{\text{U}}_{t}+\hat{R}^{\textrm{rect}}_{t}|\eta_{t}]].
(31)
Since the hyperparameter
\eta_{t}
is only a function of data observed at times
t^{\prime}=1,...,t-1
, it is independent of the data observed at time
t
, and we have
\displaystyle\mathbb{E}[\hat{R}^{\textrm{U}}_{t}+\hat{R}^{\textrm{rect}}_{t}|\eta_{t}]
\displaystyle=\eta_{t}\cdot\mathbb{E}[\ell(f(\tilde{x}_{t}),\tilde{{y}}_{t})]+\mathbb{E}[\ell(f({x}_{t}),{y}_{t})-\eta_{t}\cdot\ell(f({x}_{t}),\tilde{{y}}_{t})]
(32)
\displaystyle=\mathbb{E}[\ell(f({x}_{t}),{y}_{t})]=R_{t}.
Then, this yields
\mathbb{E}[\hat{R}^{\text{U}}_{t}+\hat{R}^{\textrm{rect}}_{t}]=\mathbb{E}_{\eta_{t}}[R_{t}]=R_{t}
.
Therefore, the equality
\mathbb{E}[\hat{\bar{R}}_{t}^{\textrm{PP}}]=t^{-1}\sum_{{t^{\prime}}=1}^{t}R_{t^{\prime}}=\bar{R}_{t}
holds. This completes the proof.
A.2
Source Upper Bound
The prediction-powered risk estimator on the source domain is given by (17), i.e.,
\displaystyle\hat{R}^{\textrm{PP}}_{0}={}
\displaystyle\underbrace{\frac{\eta_{0}}{N_{0}}\sum_{j=n_{0}+1}^{n_{0}+N_{0}}\ell\big(f(\tilde{x}_{0,j}),\tilde{y}_{0,j}\big)}_{\hat{R}^{\textrm{U}}_{0}}
\displaystyle+\underbrace{\frac{1}{n_{0}}\sum_{i=1}^{n_{0}}\ell\big(f({x}_{0,i}),y_{0,i}\big)-\frac{\eta_{0}}{n_{0}}\sum_{i=1}^{n_{0}}\ell\big(f({x}_{0,i}),\tilde{y}_{0,i}\big)}_{\hat{R}^{\textrm{rect}}_{0}}.
(33)
We partition the unlabeled set into
n_{0}
disjoint blocks, each containing
N_{0}/n_{0}
points.
The
i
-th labeled sample is paired with the unlabeled block
j\in[(i-1){N_{0}}/{n_{0}}+n_{0}+1,\;\;i{N_{0}}/{n_{0}}+n_{0}].
The block-wise estimator can be denoted as
\hat{R}^{\textrm{PP}}_{0}=\frac{1}{n_{0}}\sum_{i=1}^{n_{0}}\left(\frac{n_{0}}{N_{0}}\sum_{j=(i-1)\tfrac{N_{0}}{n_{0}}+n_{0}+1}^{\,i\tfrac{N_{0}}{n_{0}}+n_{0}}\eta_{0}\cdot\ell(f(\tilde{x}_{0,j}),\tilde{y}_{0,j})+\ell(f({x}_{0,i}),y_{0,i})-\eta_{0}\cdot\ell(f({x}_{0,i}),\tilde{y}_{0,i})\right).
(34)
For convenience, define the per-sample risk
z_{0,i}^{\textrm{PP}}=\frac{n_{0}}{N_{0}}\sum_{j=(i-1)\tfrac{N_{0}}{n_{0}}+n_{0}+1}^{\,i\tfrac{N_{0}}{n_{0}}+n_{0}}\eta_{0}\cdot\ell(f(\tilde{x}_{0,j}),\tilde{y}_{0,j})+\ell(f({x}_{0,i}),y_{0,i})-\eta_{0}\cdot\ell(f({x}_{0,i}),\tilde{y}_{0,i}).
(35)
Thus, the prediction-powered estimate (17) is given by the average
\hat{R}^{\textrm{PP}}_{0}={n_{0}}^{-1}\sum_{i=1}^{n_{0}}z_{0,i}^{\textrm{PP}}
.
Therefore, the upper bound can be obtained by the algorithms proposed in
(Waudby-Smith and Ramdas, 2024)
. By viewing
z_{0,i}^{\textrm{PP}}
as new data points, we can naturally call the Betting algorithm to get the required upper bound of source risk. Since
R^{\textrm{PP}}_{0}
is an unbiased estimate of
R_{0}
, the resulting bound is naturally valid.
A.3
Proof of Lemma
3.3
For a fixed
\hat{\bar{R}}_{t-1}^{\textrm{PP}}
, we have
\mathbb{E}\big[\bigl(\hat{R}_{t}^{\textrm{PP}}(\eta_{t})-\hat{\bar{R}}_{t-1}^{\textrm{PP}}\bigr)^{2}\big]=\operatorname{Var}\bigl(\hat{R}_{t}^{\textrm{PP}}(\eta_{t})\bigr)+\bigl(\mathbb{E}[\hat{R}_{t}^{\textrm{PP}}(\eta_{t})]-\hat{\bar{R}}_{t-1}^{\textrm{PP}}\bigr)^{2}.
(36)
Using
\mathbb{E}[\hat{R}_{t}^{\textrm{PP}}(\eta_{t})]=R_{t}
, this becomes
\mathbb{E}\big[\bigl(\hat{R}_{t}^{\textrm{PP}}(\eta_{t})-\hat{\bar{R}}_{t-1}^{\textrm{PP}}\bigr)^{2}\big]=\operatorname{Var}\bigl(\hat{R}_{t}^{\textrm{PP}}(\eta_{t})\bigr)+\bigl(R_{t}-\hat{\bar{R}}_{t-1}^{\textrm{PP}}\bigr)^{2}.
(37)
The second term depends only on
R_{t}
and
\hat{\bar{R}}_{t-1}^{\textrm{PP}}
, and is therefore
independent of the hyperparameter
\eta_{t}
. Consequently,
minimizing the expectation is
equivalent to minimizing the variance term
\operatorname{Var}(\hat{R}_{t}^{\textrm{PP}}(\eta_{t}))
.
Noting that this variance minimization problem is the same as Example 6.1 of PPI++
(Angelopoulos et al., 2023a)
, we thus directly have the following optimal solution:
\eta_{t}^{*}=\frac{\operatorname{Cov}(u_{t},\tilde{u}_{t}^{\mathrm{L}})}{(1+\frac{n_{t}}{N_{t}})\operatorname{Var}(\tilde{u}_{t}^{\mathrm{U}})},
(38)
Figure 6:
Detection performance comparison between SRM and PPRM under a stationary regime. The true risk is
p=0.5
and the threshold is
\theta=0.42
. The dashed vertical lines indicate the time to alarm, while the solid vertical lines indicate the theoretical detection time computed using the approximations in (42).
A.4
Theoretical Detection Time Comparison
We next provide a theoretical analysis to illustrate when PPRM can lead to shorter detection delay. For clarity, as shown in Figure 6 , we consider a stationary regime with binary risks, where the system is already in the harmful regime when monitoring begins, so that the distribution shift is present from
t=1
onward.
Let
u_{t}=\ell(f(x_{t}),y_{t})
denote the true risk on a labeled sample;
\tilde{u}_{t}^{\mathrm{L}}=\ell(f(x_{t}),f_{\emph{p}}(x_{t}))
and
\tilde{u}_{t}^{\mathrm{U}}=\ell(f(\tilde{x}_{t}),f_{\emph{p}}(\tilde{x}_{t}))
denote the surrogate risk on a labeled sample and on an unlabeled sample, respectively. We assume
u_{t}^{\textrm{U}}\sim\mathrm{Bernoulli}(q)
and
u_{t}^{\textrm{L}}\sim\mathrm{Bernoulli}(q)
. Let
u_{t}\sim\mathrm{Bernoulli}(p)
,
p>\theta
, with
\theta
denoting the alarm threshold, and
\Delta=p-\theta>0
the risk margin. Let
\gamma=\operatorname{Cov}(u_{t},\tilde{u}_{t}^{\mathrm{L}})
.
We further assume fixed batch sizes
n_{t}=n
and
N_{t}=N
, and use a fixed reliance parameter
\eta_{t}=\eta
.
For SRM, the estimate
\hat{R}_{t}
is the empirical mean of
n
labeled samples, so its variance is
v^{\textrm{S}}=\frac{p(1-p)}{n}.
(39)
For PPRM, the prediction-powered estimate
\hat{R}_{t}^{\textrm{PP}}(\eta)
has variance
v^{\textrm{PP}}(\eta)=\eta^{2}\frac{q(1-q)}{N}+\frac{1}{n}\left(p(1-p)+\eta^{2}q(1-q)-2\eta\gamma\right).
(40)
We consider the linear boundary associated with the sub-exponential transform
\psi_{E}(\lambda)=-\log(1-\lambda)-\lambda,\quad\lambda\in(0,1)
(
(Howard et al., 2020)
, Theorem 1)
u(v)=\frac{\log(1/\delta_{\mathrm{T}})}{\lambda}+\frac{\psi_{E}(\lambda)}{\lambda}v.
(41)
In the considered scenario,
\hat{\bar{R}}_{t}
converges almost surely to
p
. Moreover, since the variance process
V_{t}
is formed from squared deviations around a predictable running mean that also converges to
p
, we use the large-
t
approximations
\hat{\bar{R}}_{t}\approx p,\qquad V_{t}\approx tv^{\textrm{S}}.
(42)
For SRM, the lower confidence bound crosses the threshold when
p-\frac{u(V_{t})}{t}>\theta.
(43)
Substituting the linear boundary and solving for
t
gives the resulting detection delay
\tau^{\textrm{S}}(\lambda)=\frac{\log(1/\delta_{\mathrm{T}})}{\lambda\Delta-\psi_{E}(\lambda)v^{\textrm{S}}}.
(44)
For PPRM, the prediction-powered estimate
\hat{R}_{t}^{\textrm{PP}}(\eta)
lies in
[-\eta,1+\eta]
. Therefore, before applying the linear boundary (41), we affinely normalize it to
[0,1]
. Let
\hat{X}_{t}^{\textrm{PP}}=\frac{\hat{R}_{t}^{\textrm{PP}}(\eta)+\eta}{1+2\eta}\in[0,1].
(45)
On this normalized scale, the mean, threshold, and risk margin are respectively given by
\mathbb{E}[\hat{X}_{t}^{\textrm{PP}}]=\frac{p+\eta}{1+2\eta},\qquad\theta_{X}=\frac{\theta+\eta}{1+2\eta},\qquad\Delta_{X}=\mathbb{E}[\hat{X}_{t}^{\textrm{PP}}]-\theta_{X}=\frac{\Delta}{1+2\eta}.
(46)
The corresponding variance is
v_{X}^{\textrm{PP}}(\eta)=\frac{v^{\textrm{PP}}(\eta)}{(1+2\eta)^{2}}.
(47)
Thus, under the same approximation as (42), we have
\hat{\bar{X}}_{t}^{\textrm{PP}}\approx\frac{p+\eta}{1+2\eta},\qquad V_{t,X}^{\textrm{PP}}\approx t\frac{v^{\textrm{PP}}(\eta)}{(1+2\eta)^{2}}.
(48)
The alarm condition for PPRM should therefore be written on the normalized scale as
\hat{\bar{X}}_{t}^{\textrm{PP}}-\frac{u(V_{t,X}^{\textrm{PP}})}{t}>\theta_{X}.
(49)
Solving for
t
gives the approximate delay
\tau^{\textrm{PP}}(\eta)=\frac{\log(1/\delta_{\mathrm{T}})}{\lambda\Delta/(1+2\eta)-\psi_{E}(\lambda)v^{\textrm{PP}}(\eta)/(1+2\eta)^{2}}.
(50)
Our numerical results in Figure 6 show that the derived approximate detection delay is close to the empirical detection delay observed in simulations.
We can now state the resulting conclusion.
Proposition A.1
(Approximate delay improvement condition)
.
Consider the above stationary regime, where distribution shift is present from
t=1
onward. Let the true labeled risk be binary with
u_{t}\sim\mathrm{Bernoulli}(p)
and
p>\theta
, and let the predicted risk satisfy
\tilde{u}_{t}^{\mathrm{L}}\sim\mathrm{Bernoulli}(q)
with
\gamma=\operatorname{Cov}(u_{t},\tilde{u}_{t}^{\mathrm{L}})
. Assume that the unlabeled
predicted risk
\tilde{u}_{t}^{\mathrm{U}}
has the same marginal distribution as
\tilde{u}_{t}^{\mathrm{L}}
. Define the margin
\Delta=p-\theta>0
, fix the reliance parameter
\eta_{t}=\eta\geq 0
, and consider the
linear boundary (
(Howard et al., 2020)
, Theorem 1) induced by
\psi_{E}(\lambda)=-\log(1-\lambda)-\lambda
for
\lambda\in(0,1)
.
Let
v^{\mathrm{S}}
and
v^{\mathrm{PP}}(\eta)
denote the variances of the
SRM and PPRM estimators, respectively. Then the approximate deterministic crossing time of PPRM is smaller than that of SRM, i.e.,
\tau^{\mathrm{PP}}(\eta,\lambda)<\tau^{\mathrm{S}}(\lambda),
(51)
if and only if
2\eta(1+2\eta)\lambda\Delta<\psi_{E}(\lambda)\left((1+2\eta)^{2}v^{\mathrm{S}}-v^{\mathrm{PP}}(\eta)\right).
(52)
Specifically, an informative predictor can reduce the effective variance
v^{\mathrm{PP}}(\eta)
through its positive covariance with the true loss, thereby tightening the lower confidence sequence and accelerating detection. However, increasing
\eta
also enlarges the range of the prediction-powered estimate from
[0,1]
to
[-\eta,1+\eta]
, which incurs the normalization penalty
(1+2\eta)^{2}
. Thus, PPRM improves the approximate detection delay when the induced variance reduction is large enough to offset this range-normalization penalty.
Appendix B
Unsupervised Lower Confidence Sequences
Unsupervised risk monitoring has been recently proposed as a technique to track the performance of methods in scenarios where only an
unlabeled
test stream
\tilde{\mathcal{D}}_{t}=\{\tilde{x}_{{t},i}\}_{i=1}^{N_{t}}
is available. To address the lack of labels, the authors of
(Schirmer et al., 2025)
replace a sequence of supervised losses with a sequence of loss proxies that can be computed directly from the unlabeled test stream. This approach enables the derivation of an unsupervised lower bound on the running test risk, which can then be used to design an unsupervised alarm function.
The core idea is to employ a proxy function
g(\cdot)
, serving as a loss proxy for the model
f(\cdot)
, defined as
{r}_{t}=g({x}_{t},f)
.
In addition to being unsupervised (i.e., depending only on inputs {x}_{t}), the proxy should be at least partially informative about the corresponding loss variable. When
g(\cdot)
satisfies the following assumption from
(Schirmer et al., 2025)
, the running test risk can be lower bounded as follows.
Assumption B.1
(Assumption 1,
(Schirmer et al., 2025)
)
.
Given a sequence of losses
{u}_{0:t}
, let the corresponding sequence of loss proxies
{r}_{0:t}
and proxy thresholds
\beta_{0},\ldots,\beta_{t}\in\mathbb{R}
, along with a loss threshold
\tau
, be such that for all
t\geq 1
, the following inequality holds:
\frac{1}{t}\sum_{{t^{\prime}}=1}^{t}\underbrace{\mathbb{P}_{P_{t^{\prime}}}\left({r}_{t^{\prime}}>\beta_{t^{\prime}},{r}_{t^{\prime}}\leq\tau\right)}_{\textrm{PFP}_{t^{\prime}}}\leq\underbrace{\mathbb{P}_{P_{0}}\left({r}_{0}>\beta_{0},{u}_{0}\leq\tau\right)}_{\textrm{PFP}_{0}}+\frac{1}{t}\sum_{{t^{\prime}}=1}^{t}\underbrace{\mathbb{P}_{P_{t^{\prime}}}\left({r}_{t^{\prime}}\leq\beta_{t^{\prime}},{u}_{t^{\prime}}>\tau\right)}_{\textrm{PFN}_{t^{\prime}}}.
(53)
The running test risk can be lower bounded as
\bar{R}_{t}\geq\tau\!\left(\frac{1}{t}\sum_{{t^{\prime}}=1}^{t}\mathbb{P}_{P_{t^{\prime}}}\!\left({r}_{t^{\prime}}>\beta_{t^{\prime}}\right)-\mathbb{P}_{P_{0}}\!\left({r}_{0}>\beta_{0},\,{u}_{0}\leq\tau\right)\right),\quad\forall\,t\geq 1.
(54)
Specifically, the authors of
(Schirmer et al., 2025)
proposed online threshold calibration to select
\beta_{0},\ldots,\beta_{t}\in\mathbb{R}
and
\tau
by maximizing the F1 score based on the source model’s proxy.
Importantly, this bound depends only on the test loss proxies and the source loss, which means that its corresponding lower-bound confidence sequence can be evaluated using a combination of unlabeled test data
\{\tilde{\mathcal{D}}_{t^{\prime}}\}_{{t^{\prime}}=1}^{t}=\{\,\{{x}_{{t^{\prime}},i}\}_{i=1}^{N_{t^{\prime}}}\,\}_{{t^{\prime}}=1}^{t}
and labeled source data
\mathcal{D}_{0}
.
In practice, this requires computing an empirical estimate of
\mathbb{P}_{P_{t^{\prime}}}\!\left({r}_{t^{\prime}}>\beta_{t^{\prime}}\right)
at each step, followed by the construction of a lower confidence sequence relying on specific bounds.
This makes it suitable for the final proposed unsupervised alarm:
\Phi_{t}^{b}=\mathbbm{1}\left[L_{t}\left({r}_{0:t},\beta_{0:t},{u}_{0},\tau\right)>U_{0}({r}_{0})+\epsilon_{\textrm{tol}}\right].
(55)
Appendix C
Supplementary Experiments
Figure 7:
Risk estimates as a function of time
t
and average time to alarm for an image classification task: (a) under increasing shift severity (squared loss); (b) under a non-monotonic shift severity (binary loss).
Figure 8:
Performance for the LLM QA task under prompt shifts: (a) running risk lower bounds and average time to alarm; (b) average time to alarm using different predictors.
C.1
Sequential Risk Monitoring on Image Data
In Figure 7 (a), we evaluate image classification risk under increasing distribution shift severity, using the squared loss.
We observe a consistent performance gain achieved by PPRM.
In Figure 7 (b), we further consider non-monotonic distribution shift scenarios, where PPRM continues to outperform all benchmark methods.
C.2
Monitoring an LLM with Limited Human-Labeled Data
In Figures
8
(a) and (b), we report additional results for monitoring another lightweight model, namely the 4-bit quantized version of MedGemma-4B-IT
(Sellergren et al., 2025)
, which is specialized for the medical domain. For the predictor
f_{\textrm{p}}(\cdot)
, we employ the non-specialized model, Qwen2.5-VL-7B in Figure 8 (a). The evaluation begins with medical questions and gradually incorporates queries from another domain. The results are consistent with those presented in the main text, demonstrating the generalization capability of PPRM.
C.3
Detecting the Need for Channel Covariance Re-Estimation
In wireless communication, reliable communication depends on accurate knowledge of the channel statistics, particularly the channel covariance matrix that determines both the quality of
minimum mean-square error
(MMSE) estimation and the performance of downstream decoders. However, in practical wireless systems, the propagation environment evolves over time due to user mobility and temporal variations in interference. These drifts cause a mismatch between the true channel distribution and the one assumed by the receiver, leading to degraded channel estimates and increased decoding errors.
As a result, a central question is when the receiver should trigger covariance re-estimation or retraining of its model parameters. Re-estimating too frequently wastes pilot and computation resources, whereas re-estimating too late leads to significant performance loss. This motivates a principled mechanism that can continuously monitor the channel quality and reliably detect when a statistically significant covariance shift occurs.
In this subsection, we examine a communication-inspired setting where the monitored model operates over a channel whose characteristics may drift over time.
C.3.1
Problem Formulation
We consider a standard uplink communication scenario with a
base station
(BS) equipped with
M
antennas and a single-antenna
user equipment
(UE). We assume block flat-fading channels, where each coherence block contains
\tau_{c}
complex-valued samples.
We denote by
t
the index associated with a single coherence block. Each block contains
n_{f}
frames for data transmission, with each
frame consisting of a codeword encoding a different message. For each frame
i
in a coherence block
t
, the
M× 1
received
signal at the BS is given by
{v}_{t}[i,j]=h_{t}s_{t}[i,j]+{e}_{t}[i,j],\quad j=1,\ldots,n_{w}
,
where
n_{w}
is the codeword length,
s_{t}[i,j]\in\mathbb{C}
is the information-bearing signal transmitted over channel use
j
, which is drawn from an existing constellation
\mathcal{M}
,
h_{t}
is the
M× 1
channel vector, and
e_{t}[i,j]\sim\mathcal{CN}(0,\sigma^{2}{I}_{M})
represents
additive white Gaussian noise
(AWGN) with zero mean and covariance
\sigma^{2}I_{M}
.
The channel vectors are assumed to be conditionally Gaussian distributed given a set of parameters
\psi_{t}
, i.e.,
h_{t}\,|\,\psi_{t}\sim\mathcal{CN}(0,C(\psi_{t}))
. The parameter
\psi_{t}
accounts for geometric and electromagnetic properties of the propagation environment that change slowly over the coherence interval index
t
. Furthermore, the covariance matrix can be expressed as
C_{t}=C(\psi_{t})=\int_{-\pi}^{\pi}g(\theta_{t};\psi_{t})\,a(\theta_{t})a^{\textrm{H}}(\theta_{t})\,d\theta_{t},
(56)
where
g(\theta_{t};\psi_{t})\geq 0
is the power density function dictated by the parameters
\psi_{t}
and
a(\theta_{t})
denotes the array response vector at the BS for an angle of arrival
\theta_{t}
.
In order to support the estimation of the channel covariance matrix
C_{0}
and the channel noise
\sigma^{2}
, at the first coherence block,
indexed as
t=0
, the UE sends
n_{c}\leq\tau_{c}
pilots
x_{c}=[x_{c}[1],\ldots,x_{c}[n_{c}]]^{\top}\in\mathbb{C}^{n_{c}× 1}
. In order to estimate the parameters
C_{0}
and
\sigma^{2}
, we adopt the positive semidefinite least squares estimate approach introduced in
(Dietrich et al., 2006)
, and the estimated parameters are represented by
{\widehat{C}}_{0}
and
\hat{\sigma}^{2}
.
At each coherence block
t
, excluding
t=0
,
n_{h}
pilots
x_{t,h}=[x_{t,h}[1],x_{t,h}[2],\ldots,x_{t,h}[n_{h}]]^{\top}\in\mathbb{C}^{n_{h}× 1}
are transmitted to enable channel
estimation. We consider MMSE channel estimation, and thus the estimated channel vector
\hat{h}_{t}
is given by
{\hat{h}}_{t}={\widehat{C}}_{0}Z_{t,h}^{\textrm{H}}\left(Z_{t,h}{\widehat{C}}_{0}Z_{t,h}^{\textrm{H}}+\hat{\sigma}^{2}I_{Mn_{h}}\right)^{-1}{y}_{t,h},
(57)
where
Z_{t,h}=x_{t,h}\otimes I_{M}
and
{y}_{t,h}={Z}_{t,h}h_{t}+e_{t,h}
. At each block
t
, the BS carries out equalization using the estimated channel
vector
{\hat{h}}_{t}
in (57). The MMSE equalized signal
\tilde{s}_{t}[i,j]
is obtained as
\tilde{s}_{t}[i,j]=\left({\hat{h}}_{t}^{\textrm{H}}{\hat{h}}_{t}+\hat{\sigma}^{2}\right)^{-1}{\hat{h}}_{t}^{\textrm{H}}v_{t}[i,j]
.
The equalized symbols for each frame are processed by an NN-based decoder,
f(\cdot)
, to obtain an estimate of the transmitted symbol,
\hat{s}_{t}[i,j]=\arg\max_{m\in\mathcal{M}}f(\tilde{s}_{t}[i,j])_{m}
,
where
\mathcal{M}
denotes the modulation set. For risk monitoring, we additionally introduce
n_{r}
pilots,
x_{t,r}=[x_{t,r}[1],x_{t,r}[2],\ldots,x_{t,r}[n_{r}]]^{\top}\in\mathbb{C}^{n_{r}× 1}
. The equalized pilot signals are denoted as
{\tilde{x}}_{t,r}=[\tilde{x}_{t,r}[1],\tilde{x}_{t,r}[2],\ldots,\tilde{x}_{t,r}[n_{r}]]^{\top}\in\mathbb{C}^{n_{r}× 1}
. These signals are decoded by the decoder, and the decoded pilots are denoted as
{\hat{x}}_{t,r}=[\hat{x}_{t,r}[1],\hat{x}_{t,r}[2],\ldots,\hat{x}_{t,r}[n_{r}]]^{\top}\in\mathbb{C}^{n_{r}× 1}
.
At each coherence block
t\geq 1
, we collect two sets of observations:
•
A labeled set from additional pilot symbols:
\mathcal{D}_{t}=\big\{(\tilde{x}_{t,r}[k],x_{t,r}[k])\ \big|\ k=1,\ldots,n_{r}\big\},
where
x_{t,r}[k]
is the ground-truth transmitted pilot symbol.
•
An unlabeled set from data symbols:
\tilde{\mathcal{D}}_{t}=\big\{\tilde{s}_{t}[i,j]\ \big|\ i=1,\ldots,n_{f},\;j=1,\ldots,n_{w}\big\},
for which the true symbol
s_{t}[i,j]
is unknown.
Given a decoder
f(\cdot)
producing the symbol estimate
\hat{s}_{t}[i,j]=\arg\max_{m\in\mathcal{M}}f(\tilde{s}_{t}[i,j])_{m}
,
the prediction-powered risk estimator at block
t
is given by
\hat{R}_{t}^{\textrm{PP}}=\underbrace{\frac{\eta_{t}}{n_{w}n_{f}}\sum_{i=1}^{n_{f}}\sum_{j=1}^{n_{w}}\ell\!\left(f(\tilde{s}_{t}[i,j]),\hat{s}_{t}[i,j]\right)}_{\hat{R}^{\textrm{U}}_{t}}+\underbrace{\frac{1}{n_{r}}\sum_{k=1}^{n_{r}}\Big(\ell(f(\tilde{x}_{t,r}[k]),x_{t,r}[k])-\eta_{t}\cdot\ell(f(\tilde{x}_{t,r}[k]),\hat{x}_{t,r}[k])\Big)}_{\hat{R}^{\textrm{rect}}_{t}}.
(58)
Figure 9:
Risk estimates as a function of time
t
and average time to alarm for the channel equalization task: (a) under increasing
\psi_{t}
; (b) under a non-monotonic
\psi_{t}
.
C.3.2
Results
We consider a scenario for the simulation in which a BS with
16
antennas receives uplink transmissions from a single-antenna UE.
The BS antenna array is assumed to be a uniform linear array, with an inter-element spacing of half the wavelength.
The
codeword length is set to
n_{w}=16
, the number of frames per coherence block is
n_{f}=2
and the signal-to-noise ratio is
6
dB. Additionally, we set
n_{c}=500
,
n_{h}=3
, and
n_{r}=1
. For simplicity, we consider the 3GPP spatial channel model for a uniform linear array with only a single propagation path. In this case, we have only one parameter for the covariance matrix: the angle of
the path center
\psi_{t}\in[-\pi/2,\pi/2]
, which is uniformly distributed. The power density function of the angle of arrival is given
by
g(\theta_{t};\psi_{t})=\exp\!\left(-\frac{d_{2\pi}(\theta_{t},\psi_{t})}{\sigma_{\textrm{ASD}}}\right),
(59)
where
d_{2\pi}(\theta_{t},\psi_{t})
is the wrap-around distance between
\theta_{t}
and
\psi_{t}
, which can be thought of as
|\theta_{t}-\psi_{t}|
for most
(\theta_{t},\psi_{t})
pairs.
In addition,
\sigma_{\textrm{ASD}}
is the angular standard deviation, as it determines how large the deviations from the nominal angle are. We gradually change the parameter
\psi_{t}
to simulate an increase in risk. Besides, squared loss is selected as the risk function.
Here we investigate a self-synthetic setting, where no additional models are needed to generate the synthetic labels. The results in Figure 9 demonstrate that PPRM can achieve earlier detection on average compared to SRM. These results highlight PPRM’s suitability for on-device model maintenance, where quick adaptation decisions are critical under bandwidth and labeling constraints.
Appendix D
Additional Analysis
D.1
Effect of Window Length
We analyze how the choice of the window length
L
affects the performance of PPRM in terms of risk monitoring and detection delay. The results in Figure 10 show that PPRM is relatively robust to the choice of
L
. In addition, the results in the right panel show a clear trade-off. Smaller windows are better for abrupt changes, whereas larger windows exhibit better performance for gradual changes. Across most window sizes, PPRM achieves a shorter average time to alarm than the SRM baselines. These results suggest that a moderate window length offers a good balance between detection sensitivity and stability. According to these results, we select
L=60
for its good performance.
Figure 10:
Comparison of running risk and average time to alarm under abrupt and gradual change scenarios for an image classification task.
D.2
Monitoring the Risk under Abrupt Shifts
Figure 11:
Risk estimates as a function of time
t
under abrupt shifts: (a) for an image classification task; (b) for an LLM QA task.
In Figure 11 , we evaluate the performance of PPRM under abrupt distribution shifts, where the risk suddenly increases at a certain time step. We observe that PPRM can quickly detect the shift and raise an alarm.
D.3
Effect of Predictor Accuracy
We further analyze how the accuracy of the predictor affects the performance of PPRM in terms of risk monitoring and detection delay. The results in Figure 12 show that as the accuracy of the predictor increases, PPRM achieves a tighter confidence sequence and faster detection compared to SRM. When the predictor is inaccurate, the adaptive version of PPRM performs comparably to SRM, while the non-adaptive PPRM may even underperform SRM due to the use of a poorly calibrated predictor. This demonstrates the benefit of the adaptive mechanism, which prevents PPRM from relying too heavily on an uninformative predictor. Moreover, in the right column of Figure 12 , we also report the curve of the obtained
\eta_{t}
, which is selected by optimizing the variance reduction at each time step. We observe that the optimized
\eta_{t}
tends to be larger when the predictor is more accurate.
Figure 12:
Results under increasing shift severity for an image classification task, using predictors of different accuracies.
Figure 13:
Results under increasing shift severity for an image classification task, using different predictors.
In Figure 13 , we report additional results for the image classification task, using predictors of different accuracies. The results show that PPRM can effectively leverage predictors of varying quality, where a more accurate predictor leads to a tighter confidence sequence and faster detection. Moreover, we also report the curve of cumulative squared prediction error
V_{t}^{\textrm{PP}}
in the right panel of Figure 13 . From the result, we can see that the resulting
V_{t}^{\textrm{PP}}
of an accurate predictor is smaller than that of a less accurate predictor, which is consistent with the variance reduction analysis in Appendix
A.3
.
Figure 14:
Risk estimates as a function of time
t
under increasing shift severity for an image classification task, in the intermittent label stream setting.
D.4
Intermittent Label Streams
In this section, we extend our analysis to the setting where labels are only observed at a subset of time steps, and evaluate the performance of PPRM under this intermittent label stream scenario. Specifically, we consider a setting where labels are observed at every
k
-th time step, i.e., at time steps
t\in\{k,2k,3k,\ldots\}
, while at other time steps, only unlabeled data are available. We adapt the PPRM estimator to this setting by only updating the risk estimate at time steps where labels are observed, while using the prediction-powered estimates at all time steps.
We now provide a simple derivation for the intermittent-label setting, in which labels are observed only every
k
time steps. Let
\mathcal{L}_{t}=\{k,2k,\ldots,\lfloor t/k\rfloor k\}
denote the set of labeled time steps up to time
t
, and let
\mathcal{U}_{t}=\{1,\ldots,t\}\setminus\mathcal{L}_{t}
denote the unlabeled ones. Then, the running risk can be decomposed as
\bar{R}_{t}=\frac{1}{t}\sum_{s=1}^{t}R_{s}=\frac{1}{t}\sum_{s\in\mathcal{L}_{t}}R_{s}+\frac{1}{t}\sum_{s\in\mathcal{U}_{t}}R_{s}.
(60)
For unlabeled time steps
s\in\mathcal{U}_{t}
, we use the unsupervised lower-bound idea from Appendix
B
. Specifically, summing over all unlabeled time steps gives
\frac{1}{t}\sum_{s\in\mathcal{U}_{t}}R_{s}\geq\frac{\tau}{t}\left(\sum_{\begin{subarray}{c}s=1\\
s\bmod k\neq 0\end{subarray}}^{t}\mathbb{P}(r_{s}>\beta_{s})-\bigl(t-\lfloor t/k\rfloor\bigr)\mathbb{P}_{P_{0}}\!\left({r}_{0}>\beta_{0},\,{u}_{0}\leq\tau\right)\right),
(61)
where
\beta_{s}
and
\tau
are the proxy thresholds for the unlabeled time steps, which can be selected by maximizing the F1 score based on the source model’s proxy, as in
(Schirmer et al., 2025)
.
Combining the labeled and unlabeled terms, we get the following simplified lower bound for the running risk
\bar{R}_{t}\geq\frac{1}{t}\sum_{j=1}^{\lfloor t/k\rfloor}R_{jk}+\frac{\tau}{t}\left(\sum_{\begin{subarray}{c}s=1\\
s\bmod k\neq 0\end{subarray}}^{t}\mathbb{P}(r_{s}>\beta_{s})-\bigl(t-\lfloor t/k\rfloor\bigr)\mathbb{P}_{P_{0}}\!\left({r}_{0}>\beta_{0},\,{u}_{0}\leq\tau\right)\right).
(62)
Then, we can construct a lower confidence sequence for the running risk by applying the PPRM estimator to the labeled time steps and using the approximation in
(Schirmer et al., 2025)
to the unlabeled time steps.
This expression also makes explicit that the intermittent-label setting interpolates between PPRM and unsupervised risk monitoring. When
k=1
, all time steps are labeled and (62) reduces to the fully labeled/PPRM monitoring setting after replacing the risks by their PPRM estimates. In contrast, as
k\to\infty
, labeled observations disappear and the bound reduces to a purely unsupervised lower bound based on the proxy scores.
We evaluate the performance of PPRM under this intermittent label stream setting, and compare it with the standard SRM method that only updates the risk estimate at time steps where labels are observed. The results in Figure 14 show that PPRM can still effectively leverage the additional predictors to achieve faster detection compared to SRM.
D.5
Cost Analysis
We further discuss the additional cost introduced by PPRM compared with SRM. One key component of PPRM is the estimation of the adaptive coefficient, which can be computed using simple empirical covariance and variance statistics from observed samples. This estimation is lightweight and incurs negligible overhead. In addition, the auxiliary predictor required by PPRM is flexible and does not necessarily introduce a heavy computational burden. It can be implemented as a lightweight model for efficient online deployment, as a stronger offline model when additional computation is available, or even as the deployed model itself, as demonstrated in our experiments.
