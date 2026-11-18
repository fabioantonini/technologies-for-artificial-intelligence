---
title: "What Carries Forward"
subtitle: "A bridge from Technologies for Artificial Intelligence to Toolkit for Modern Algorithms"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Not examinable · reading time about 30 minutes"
---

# What carries forward

This course covered no large language models, no transformers, no retrieval and
no agents. That was deliberate: they belong to **Toolkit for Modern Algorithms**,
which follows this one in the programme and is taught by someone else.

This document is not about them either. It is about **what you already own**,
arranged by what it turns into over there.

## Why it is worth thirty minutes

There is a predictable mistake waiting at the start of the next course, and it
is the same shape as the ones this course kept naming.

A language model looks like a different subject. It has its own vocabulary, its
own tooling, its own conferences, and demonstrations that work on the first try.
The reasonable conclusion is that the ten weeks behind you were the *previous*
subject, and that the new one starts from zero.

The consequence is specific and common: **you will be able to build things you
cannot evaluate.** Getting a retrieval system to answer a question takes an
afternoon. Knowing whether it answers better than the string-matching baseline
it replaced, whether the benchmark it scores 82% on contains its own training
data, and whether 82% is even different from 79% given how few questions were
asked — that is this course, and nothing in the next one will teach it again
because it is assumed.

Lesson 10 closed on the sentence this whole document turns on:

> A model turns a representation into a decision. **The representation usually
> matters more than the model.** Classical methods make you build it;
> convolutional networks learn it, paying in data and compute for the privilege.
> Neither is a default.

A language model is a representation engine of enormous generality. That
changes what is easy. It changes nothing about how you decide whether a
representation is any good.

---

## 1. Five things that are not analogies

Start here, because these are not "similar to" what you learned. They are the
same objects under different names, and recognising them saves you learning
anything twice.

### 1.1 Predicting the next token is lesson 4's classifier

Lesson 4 derived cross-entropy from maximum likelihood and made a point of
saying it is **not a design choice** — it is what maximum likelihood gives you
once you assume a Bernoulli outcome. That lesson's section 10 generalised it to more
than two classes as **categorical cross-entropy**: the negative log of the
probability the model gave to what actually happened. Section 10 and its slides
60 to 62 were optional in class; read them before the next course, because
everything in this section starts there.

Next-token prediction is that, with the classes being the vocabulary. The loss a
language model is trained on is the loss you derived in week four, over a larger
set of outcomes. Nothing else about it is new.

**The number they report: perplexity.** Language models are compared by their
*perplexity*, and it is lesson 4's log loss with one more step. With $J$ the
average log loss in natural logarithms, as lesson 4 and scikit-learn's
`log_loss` compute it:

$$\text{perplexity} = e^{J}$$

The rule that makes it readable: a model that spreads its probability evenly
over $K$ outcomes gives each one $1/K$, so every example costs
$-\ln(1/K) = \ln K$, and the perplexity is $e^{\ln K} = K$. **Perplexity is the
number of equally likely outcomes the model is as unsure as.** Equivalently,
$e^{-J}$ is the probability the model gave to what happened, averaged
geometrically over the examples.

Lesson 4's drives, on the 6,000 training drives of notebook 1, of which 230 fail.
A constant prediction $p$ on a set where a share $q$ fails costs
$q \times (-\ln p) + (1 - q) \times (-\ln(1 - p))$ per drive:

| Model | Average log loss $J$ | Perplexity $e^J$ | What it means |
|---|---|---|---|
| $w = 0$, $b = 0$: every drive gets $p = 0.5$ | $-\ln 0.5 = \ln 2 = 0.6931$ | $e^{0.6931} = 2.00$ | a coin flip: two equally likely outcomes |
| every drive gets the base rate, $p = q = 230 \div 6{,}000 = 0.03833$ | $0.03833 \times 3.2614 + 0.96167 \times 0.03909 = 0.1250 + 0.0376 = 0.1626$ | $e^{0.1626} = 1.18$ | knows only how rare failures are |
| gradient descent, 4,000 iterations (notebook 1) | $0.0692$ | $e^{0.0692} = 1.07$ | gives what happened $e^{-0.0692} = 0.933$ on (geometric) average |

The first row is the "free test" of lesson 4's section 3.4: any model that starts
from zero weights must start at a perplexity of exactly 2. The last row shows the
trap lesson 4 was built around, in a new unit. A perplexity of 1.07 sounds close
to certainty, but a model that knows nothing except the base rate already scores
1.18: like 96.20% accuracy, the number means something only against what knowing
nothing scores (section 2.4 below).

For a language model the outcomes are the tokens of its vocabulary. With an
illustrative vocabulary of 50,000 tokens, a model that guesses uniformly has
$J = \ln 50{,}000 = 10.82$ and a perplexity of 50,000. A model with $J = 2$ has a
perplexity of $e^2 = 7.39$: at each step it is as unsure as if it were choosing
among about seven equally likely tokens out of 50,000.

One condition, and it is lesson 5's: two perplexities can be compared only on the
same text, split into tokens the same way. A model with a different vocabulary
predicts a different number of tokens, each with a different difficulty, so its
perplexity is measured on a different test set, and the comparison is no
comparison at all.

### 1.2 The softmax is the one from lessons 4 and 9

$$\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Lesson 4 section 10.1 introduced it on drives with three classes (healthy,
degraded, failed) and showed that with two classes it is the sigmoid: subtract
$z_0$ from both scores and the probability of class 1 becomes
$\sigma(z_1 - z_0)$. The binary logistic regression of that lesson simply fixes
$z_0 = 0$, so its single score $z$ is already the difference, the log-odds.
Lesson 9 section 5.3 wrote the softmax down again and trained ten-class networks
with it.
Paired with cross-entropy its gradient is again $\hat{y} - y$, the blame section 4.3
derived for one sigmoid unit; the advanced version of the lesson, in `Advanced/`,
derives the ten-class case in full.
It is the last layer of a language model, and the *temperature* you will meet as
a sampling knob is the divisor inside the exponent.

### 1.3 Retrieval is lesson 6's k-nearest neighbours

Fetching relevant documents before answering is k-nearest neighbours over
learned representations. The vocabulary changes — *embedding*, *vector store*,
*top-k* — and the algorithm does not. Four things carry over exactly.

**The index is the fit, the query is the predict.** Lesson 6's `fit` computed no
distance: it stored the pumps, at most organised in a tree so that later
searches could skip whole regions. Building a vector store's index is that step;
answering a query is `predict`, where all the work is. The same asymmetry, the
same cost profile: cheap to build, paid for at every question.

**Retrieval is the search without the vote.** k-NN finds the $k$ nearest and lets
their labels vote; retrieval returns the $k$ nearest themselves. And the metric
is a modelling decision rather than a default, as lesson 6 said of scaling -
with one simplification worth knowing: for vectors normalised to length 1,
$\lVert u - v \rVert^2 = 2 - 2\cos(u, v)$, so Euclidean distance and cosine
similarity put the neighbours in exactly the same order.

**The curse applies, with its condition.** The measurement to remember:

> In one hundred dimensions the nearest point is **70% as far away as the
> farthest**.

That number is for columns that behave like independent random numbers - which
is what lesson 6's noise columns were, and what an embedding is not. Embeddings
have 768, 1536 or 3072 columns, but those columns move together: the vectors
occupy a region of far lower effective dimension, and lesson 6's handout
(section 3.2) shows a hundred columns driven by two hidden variables behaving
exactly like two. That is why nearest-neighbour search over embeddings works at
all. The 70% is what happens to the dimensions that are noise - so the quality
of the representation matters more than its size.

**Approximate means measured.** At millions of vectors the exact search is too
slow, and vector stores use approximate nearest-neighbour indexes: navigable
small-world graphs (HNSW), inverted files over clusters (IVF), product
quantisation, in libraries such as FAISS.
`Lessons/06_knn_naive_bayes_svm/Resources/` carries the engineering from first
principles. They trade exactness for speed, and the loss is measured, not
assumed: on a sample of queries, the share of the true $k$ nearest that the
approximate search returns, its *recall@k*. That is lesson 5's habit applied to
a data structure.

### 1.4 Sampling an answer several times and voting is lesson 7's bagging

You will meet it as *self-consistency*. Lesson 7 derived why averaging reduces
variance and, more usefully, why it stops:

$$\text{variance of the average} \longrightarrow \rho\sigma^2$$

The floor is set by how much the members agree with each other. Ten samples from
one model at one temperature are highly correlated, so $\rho$ is large and the
floor is high. That is the same argument that told you why a random forest
decorrelates its trees deliberately, and it predicts exactly where sampling more
stops helping.

### 1.5 An embedding is lesson 8's dimensionality reduction

Lesson 8 built principal component analysis as the direction with the widest
shadow and used reconstruction error to find anomalies that were invisible in
any single column. An embedding
is a learned, non-linear version of the same idea: a low-dimensional
representation in which nearness means something.

Which carries the same consequences. Clustering embeddings to discover topics is
lesson 8's k-means, with lesson 8's problem — **no labels to check against**, so
silhouette tells you a clustering is self-consistent and nothing about whether
it is right. Detecting out-of-distribution inputs by reconstruction error is
lesson 8's section 5, unchanged.

---

## 2. The methodology, which is the part nobody re-teaches

If you keep one section of this document, keep this one. The identities above
are pleasant; the following is what separates people who can evaluate a system
from people who can only demonstrate one.

### 2.1 Trying many prompts and reporting the best

Lesson 5 measured it on a table with **no signal in it whatsoever** — the true
area under the curve was **0.500**, by construction. A grid of 25 candidate
configurations, each scored on the same split, reported a best of **0.7999**.
Nested cross-validation, which refits the selection inside every fold, gave
**0.6699** on the identical data.

The gap between those two numbers is what selecting on the data you report
costs. It is not a small correction: one procedure says the model is good and
the other says there was nothing there.

Prompt engineering is model selection. Every prompt you try is a candidate;
every candidate you evaluate on the same small set of examples buys you another
chance to be fooled. `Resources/the_replication_crisis.md` from lesson 5 names
the general form — Gelman and Loken's **garden of forking paths** — and it is
the single most transferable idea in this course.

### 2.2 One split is a lottery

Lesson 5's measurement: with the selection correctly inside the pipeline, the
estimate over twenty fold seeds settles at **0.658** — but the individual splits
range from **0.576 to 0.759**. One split can hand you either end, and it will
not tell you which one it handed you.

Now add that language models are **stochastic at inference**. You have two
sources of variation where this course had one, and the discipline you learned —
repeat, report the spread, never rank on a single run — matters more, not less.

Lesson 7 taught the same lesson from the other side. Two scores agreeing to four
decimal places looked like a law and turned out to be one seed's luck: across
twelve seeds the typical gap was 0.0026. If you find yourself writing
*exactly*, run it again.

### 2.3 Benchmark contamination is leakage

Lesson 2 built a leak that raised a score, broke no rule visibly, and threw no
error — an imputer fitted before the split, so that three quarters of the
affected training rows had a test row among the five neighbours it consulted.
Lesson 5 gave it its name.

The same failure, at the scale of the internet, is a benchmark whose questions
are in the training corpus. It is the dominant methodological problem of the
field you are entering, it is *your* lesson 2 and lesson 5, and it is why a
headline benchmark number deserves the question **"where did the test set come
from, and could the model have seen it?"** before any other question.

### 2.4 A score without a baseline is not a result

Lesson 1's first measurement was a classifier that answers "negative" every
time, scores 0.986, and finds nothing. The trained model beside it scored 0.987.

"The model gets 71% on our evaluation" is not a result until you know what
answering *C* every time gets, what the previous system got, and what a
competent human gets. This is the cheapest habit in the course and the one most
often skipped when the system is impressive.

---

## 3. Your own numbers, pointed forward

| From | Number | What it warns you about there |
|---|---|---|
| Lesson 6 | nearest is **70%** as far as farthest at 100 dimensions | retrieval over embeddings: what their dimensions do to "nearest" when they carry noise rather than signal |
| Lesson 7 | **54%** of a forest's importance landed on pure noise while accuracy degraded gracefully | reading attention weights as explanations |
| Lesson 9 | six sigmoid layers never left chance, **0.1000**, where the same network with ReLU reached 0.9611 | why residual connections and normalisation exist at all |
| Lesson 4 | a false alarm costs €140, a miss €2,600 | every guardrail, filter and router is a threshold on a cost |
| Lesson 8 | **37 of 40** planted anomalies caught, and 32–38 across other draws | quoting one evaluation run as a specification |
| Lesson 10 | a random forest on raw pixels came within **two thousandths** of a convolutional network | assuming the newest architecture wins |

The lesson 7 entry deserves a sentence of its own. Its forest kept a respectable
accuracy while more than half of the importance it reported had landed on
columns that were pure noise — **the ranking was wrong in a way the accuracy
gave no hint of.** Attention weights are the same kind of artefact: a number the
model produces, which is not the same as a reason the model acted.

---

## 4. Machinery you have already built

Two of the next course's components are things you wrote yourself. Open them
again and read them as what they become — this is cheaper than any tutorial.

- **`Lessons/06_knn_naive_bayes_svm/Notebooks/01_knn_and_the_curse.ipynb`** —
  you implemented nearest-neighbour search in about eight lines and then watched
  it degrade as dimensions were added. That is a retrieval system and its
  failure mode, built by you.
- **`Lessons/08_unsupervised_learning/Notebooks/03_pca_and_anomaly_detection.ipynb`**
  — you reduced eight columns to three, reconstructed, and used the error to
  find anomalies invisible in any single column. That is the embedding-space
  machinery, minus the learning.
- **`Lessons/09_neural_networks/Notebooks/01_backpropagation_from_scratch.ipynb`**
  — you wrote backpropagation and checked it against finite differences. A transformer is
  those layers with a different mixing operation between them.

---

## 5. Where the connection is weaker than it looks

An honest bridge says where it stops.

**Scale changes which problems exist.** This course fitted models on thousands of
rows on a laptop with no graphics card. At the scale of a language model, the
failures that dominate — distributed training, memory, throughput, inference
cost — are engineering problems this course never posed.

**Some of what you learned about interpretability does not transfer cleanly.** A
decision tree's path is an explanation. Lesson 7's feature importances were
already a caution; the next course's equivalents are weaker still, and lesson 7's
Resources document on GDPR Article 22 and the right to an explanation is the
right place to have your expectations set.

**Evaluation without ground truth is genuinely harder there.** Lesson 8 said
that clustering has no answer key and that internal metrics measure
self-consistency rather than correctness. A great deal of language-model
evaluation is in that position, and the honest answer is that the field has not
solved it. What lesson 8 gives you is the reflex to notice when you are in that
position rather than to assume you are not.

---

## What to take from it

**You are not starting a new subject.** You are meeting the same objects with
different names, plus one genuinely new architectural idea.

**The methodology is not assumed knowledge that you may skip — it is the part
that will distinguish you.** A great many people can make a language model
produce something impressive. Rather fewer can say whether it is better than
what it replaced, and defend the claim.

**And the question this whole course kept asking is the one to carry across
unchanged.** Not *is this number correct* — but:

> **Does this sentence claim more than this number supports?**

---

## Where to look next

| Source | Type | Why |
|---|---|---|
| `Lessons/05_experimental_methodology/Resources/the_replication_crisis.md` | this course | The garden of forking paths, and why the problem is not one of beginners |
| `Lessons/06_knn_naive_bayes_svm/Resources/finding_neighbours_at_scale.md` | this course | Approximate nearest-neighbour search: the vector-database industry, from first principles |
| `Lessons/07_trees_and_ensembles/Resources/right_to_explanation.md` | this course | GDPR Article 22 and what an explanation is legally required to be |
| Gelman and Loken, *The Garden of Forking Paths* (2013) | paper | The original statement of the multiple-comparisons problem without explicit multiple comparisons |
| Leskovec, Rajaraman and Ullman, *Mining of Massive Datasets*, chapter 3 | book, free online | Similarity search at scale, derived rather than described |
