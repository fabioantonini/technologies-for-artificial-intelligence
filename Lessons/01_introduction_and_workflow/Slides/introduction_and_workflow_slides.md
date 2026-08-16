---
title: "Lesson 1 — Introduction and the ML Workflow"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "25 September 2026"
---

# Welcome

- 10 lessons, 3 hours each, every Friday until 27 November
- A first course in **machine learning**, for people who can already program
- Foundations, not products

::: notes
Introduce yourself briefly. Establish the two facts that matter: this is a
fundamentals course, and it assumes programming ability but no ML background.
Ask for a show of hands: who has trained a model before? Usually a handful. Tell them
that group has an advantage of about two lessons and a disadvantage that lasts longer,
because they will have to unlearn some habits around evaluation.
:::

# What we will cover

| | |
|---|---|
| 1–2 | Workflow, data preparation |
| 3–4 | Regression, classification |
| **5** | **Experimental methodology** |
| 6–7 | k-NN, SVM, trees, ensembles |
| 8 | Unsupervised learning |
| 9–10 | Neural networks, CNNs |

::: notes
Point at lesson 5 and say it is deliberately in the middle, not at the end. That is the
lesson about whether your results mean anything, and everything after it depends on it.
Mention explicitly: no LLMs in this course, they belong to another module. Some will be
disappointed; tell them the material here is what those systems are built on.
:::

# What this course is not

- Not a tour of current AI products
- Not a library tutorial
- Not a course where you call `.fit()` and report the number

::: notes
Be direct here. They can learn scikit-learn's API from documentation in an afternoon.
What takes a semester is knowing which number to trust. If they leave able to spot a
broken evaluation in someone else's work, the course succeeded.
:::

# Your advantage, and your gap

**Advantage:** you can program. You will implement methods, not only call them.

**Gap:** you can produce results faster than you can judge them.

::: notes
This is the framing for the whole course, so spend a minute on it. Within three weeks
they will write thirty lines that give 95% accuracy on something. The hard question is
whether it means anything. Say plainly: the most common failure in this field is not a
bug, it is a number that was never valid in the first place.
:::

# How you will be assessed

- **Weekly exercises** — set every Friday, due the next
- **Final project** — an end-to-end study, with peer review
- **Written exam** — closed book, drawn from the handouts

::: notes
Set expectations now. The exercises are not optional and they build towards the
project. Point them at Assessment/ in the repository, and say the project brief is
available today so they can start thinking about a dataset. Emphasise: methodology
outweighs accuracy in every component. Two students with different scores can both get
full marks; a spectacular score obtained by leaking the test set will not.
:::

# The three things you get each lesson

- **Handout** — the reference text, with the mathematics
- **Slides** — what we do in the room
- **Notebooks** — runnable implementations

Plus a quiz, and the exercise.

::: notes
Explain the division of labour, because it is unusual and it matters. Derivations are
NOT on the slides — they are in the handout, to be read afterwards. In the room we do
results, intuition and code. Tell them the exam draws on the handouts, so treating them
as optional is a mistake.
:::

# Getting the material

```bash
git clone https://github.com/fabioantonini/\
technologies-for-artificial-intelligence.git
cd technologies-for-artificial-intelligence
docker compose up
```

Then `http://127.0.0.1:8888/lab?token=aicourse`

::: notes
Do this live. Take ten minutes and make sure everyone has JupyterLab running before
moving on - it is far cheaper to fix now than during lesson 3. New material arrives with
`git pull`, a few megabytes, not a new image. No API keys, no cloud accounts, no GPU:
everything in this course runs on a laptop CPU, offline.
:::

# Today

- What learning from data means
- A short history, and what it teaches
- The three kinds of learning
- The end-to-end workflow
- Four ways a model misleads you

::: notes
Agenda slide. Flag the last item as the one that matters most today: everything before
it is orientation, and that section is where the course's real subject appears.
:::

# A problem you cannot specify

Write a program that decides whether an email is spam.

- Filter certain words? Spam adapts.
- Filter certain senders? They change.
- The exceptions grow until nobody can maintain it

::: notes
Work this example conversationally. Ask the room for rules and write two or three on the
board, then break each one. Get them to feel the frustration before offering the
alternative. The point to land: the program is not hard to write, the RULE is impossible
to state.
:::

# The inversion

You cannot state the rule.
But you can **recognise** the answer.

So: supply examples, and let a procedure find a rule consistent with them.

::: notes
This is the conceptual move that defines the field, so say it slowly. We stop
specifying behaviour and start specifying examples of correct behaviour. Everything
else in the course is machinery for doing that search well.
:::

# Making it precise

- Inputs from a space `X`, outputs from a space `Y`
- Pairs (x, y) drawn from an unknown distribution `D`
- A loss `L(ŷ, y)`: the cost of answering ŷ when the truth is y
- Goal: find f: X → Y that predicts well

::: notes
Keep this brisk - the full treatment is in handout section 2.1. The one thing to
stress is that D is UNKNOWN and FIXED. Unknown is why we need data; fixed is an
assumption that quietly fails in practice, and when it does, models degrade. Promise
them we return to that.
:::

# What we actually want

The average loss over all data the world might produce — including data that does not
exist yet:

$$R(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ L(f(x), y) \right]$$

::: notes
Name it: expected risk. Then deliver the punchline - we cannot compute it, because D is
unknown. Pause there. This is the central obstacle of the field and everything else
follows from it.
:::

# What we can actually compute

Average loss over the finite sample we happen to hold:

$$\hat{R}_S(f) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i)$$

::: notes
Empirical risk. Minimising it is called empirical risk minimisation, and essentially
every method in this course is an instance of it. Ask the room: what could go wrong
with minimising this instead of the thing we actually want? Someone will get it.
:::

# The gap that explains everything

- We **minimise** the empirical risk
- We **care about** the expected risk
- A flexible enough model drives the first to zero by memorising

That gap is **overfitting**

::: notes
The central slide of the lesson. A model that stores every training pair and recites
answers has zero empirical risk and is useless. So performance on data you fitted to is
always optimistic and sometimes meaningless. Everything about held-out data exists to
estimate the number we cannot compute.
:::

# Which gives us the one rule for today

Hold out part of the data.
Never let the fitting procedure touch it.
Measure there.

::: notes
Say clearly: this is not bureaucratic caution, it is the only thing standing between
them and a meaningless number. We will violate it deliberately later today and watch
77% accuracy appear out of pure noise.
:::

# When *not* to use machine learning

- The rule is **known** — write it
- The data is **not representative** of deployment
- Errors are **catastrophic and unexplainable**
- The data encodes an **injustice** you would automate
- A **simple baseline** already suffices

::: notes
Unfashionable and immediately useful. Give the even-number example for the first, and
say it is violated constantly because ML is the interesting option. For the fourth,
note that a model is not neutral because it is mathematical - it is a compressed summary
of the decisions in its training data. We return to this at the end.
:::

# 1943 — the first artificial neuron

McCulloch and Pitts: a neuron as a threshold unit.

Fires when the weighted sum of inputs passes a threshold.

::: notes
Start the history section. The claim was philosophical as much as technical: thought
might be computation. Weights were set by hand, nothing learned yet. Tell them the unit
in lesson 9 is this unit with a smoother threshold - the history is not decoration, it
is the same objects.
:::

# 1950 — Turing's imitation game

"Can machines think?" — replaced by a question you can test by observation.

::: notes
Highlight the MOVE, not the test. Turing replaced an unanswerable question with a
measurable proxy. That is exactly what they do every time they choose a metric, and it
carries the same obligation: stay aware of the gap between what you measure and what
you mean. Recommend the paper - it is readable and funny.
:::

# 1956 — Dartmouth

The term *artificial intelligence* is coined.

The proposal: ten people, two months, significant progress.

::: notes
Let the room react to "two months". Seventy years later it is unsolved. The optimism is
not incidental - it is present at the field's naming and it recurs on a cycle.
:::

# 1957 — the perceptron learns

Rosenblatt: weights adjusted **from examples**, not set by hand.

Convergence theorem: if the data is linearly separable, it will find a separator.

::: notes
The first machine that learns. Stress the CONDITION on the theorem - "if linearly
separable". The theorem is true and the condition is doing heavy lifting. Then contrast
with the press: the New York Times reported the Navy expected a machine that would walk,
talk, see, write and reproduce itself.
:::

# 1969 — the first winter

Minsky and Papert: a single-layer perceptron cannot represent XOR.

Correct. And narrower than the reception suggested.

::: notes
Draw XOR on the board and let them see no line separates it. Then make the key point:
the result applies to ONE LAYER. Multi-layer networks were known to be more expressive;
what was missing was a way to train them. That distinction was lost, funding collapsed,
and the field lost roughly fifteen years. In lesson 9 a hidden layer disposes of XOR in
four lines.
:::

# 1986 — backpropagation

An efficient way to train multi-layer networks.

The 1969 objection is answered. A second winter follows anyway.

::: notes
Note the technique had been derived before - Linnainmaa 1970, Werbos 1974 - without
taking hold. Ideas need context as much as correctness. The second winter came in the
early 90s: too little data, too little compute, and expert-system companies collapsing.
:::

# 1995–2010 — statistics takes over

- Support vector machines, ensembles, probabilistic methods
- From "simulating intelligence" to "estimating functions from data"

A retreat in ambition. An advance in rigour.

::: notes
Say that most of this course lives here, and that it is where the reliable tools come
from. It is unglamorous. It is also what actually gets deployed. Lessons 5, 6 and 7 are
this period.
:::

# 2012 — AlexNet

ImageNet error: 26% → 15%.

The architecture was recognisably LeNet, from 1989.

::: notes
The most important slide in the history section. What changed was DATA (1.2M labelled
images) and COMPUTE (two consumer GPUs), plus details like ReLU and dropout. A
conceptual advance would have been visible in 1989. This was a scaling result - and
saying so is not a criticism, it is the key fact about the last decade.
:::

# The pattern

| Advance | Overclaim | Correction |
|---|---|---|
| Perceptron learns | Machines that walk and talk | XOR, first winter |
| Backprop trains depth | Networks solve everything | No data, second winter |
| Scale delivers | ? | ? |

::: notes
Leave the question marks. Ask the room what belongs there. The professional skill is
distinguishing a demonstrated result from an extrapolated promise - Rosenblatt's theorem
was true, "it will reproduce itself" was not, and both appeared in the same press cycle.
Point them at Resources/history_of_ai.md for the full version with primary sources.
:::

# Three kinds of learning

The taxonomy divides by **where the target comes from** — not by algorithm.

::: notes
Correct the common misconception immediately: it is not about which algorithm you use.
The same estimator can serve more than one setting, as they will see in notebook 02.
:::

# Supervised

Each example carries a label a person produced.

- Category → **classification**
- Continuous → **regression**
- You can check answers against truth

::: notes
Lessons 3, 4, 6 and 7 are all supervised. Ask what the expensive part is - someone will
say the labels. Confirm it: in practice labels, not algorithms and not compute, are what
limits most real projects.
:::

# Unsupervised

No targets at all. Find structure.

- Groups → clustering
- Fewer dimensions → dimensionality reduction
- Odd points → anomaly detection

::: notes
Lesson 8. The difficulty to plant now: there is no accuracy to report. The algorithm
always returns groups, whether or not the data has any. Whether they mean something is
a judgement about the domain and cannot be delegated to a metric.
:::

# Self-supervised

Hide part of the input. Predict it from the rest.

Nobody annotates anything — and the supervision is real.

::: notes
The one they will have heard about without a clean definition. The reconstruction task
is a PRETEXT: nobody wants a flavanoid predictor. Solving it forces the model to
represent how the parts relate, and that transfers. Scale it up - hide a word, hide an
image patch - and it is how modern large models are trained. Free supervision is why
they scale.
:::

# One dataset, three questions

![](kinds_of_learning.png)

::: notes
Walk through notebook 02 here, live. Same 178 wines in all three panels. Left: what the
algorithm sees without labels. Middle: labels a person supplied. Right: groups found by
geometry alone. Point out the cluster colours do not match the label colours - k-means
numbers arbitrarily, and without labels there is nothing to align to. Warn them the high
agreement here is a happy accident of this dataset.
:::

# Where the target comes from

| | Target | Cost | Measurable? |
|---|---|---|---|
| Supervised | A person | Expensive | Yes |
| Unsupervised | None | Free | Not directly |
| Self-supervised | The input | Free | Only the pretext |

::: notes
Read the cost column: it explains why self-supervision dominates where data is abundant
and annotation is not. Then read the right-hand column and flag the trap: in two of
three cases you cannot straightforwardly measure success at the thing you want.
:::

# The workflow

1. Frame the problem
2. Inspect the data
3. **Split**
4. Baseline
5. Preprocess and model, in a pipeline
6. Evaluate with an honest metric
7. Diagnose
8. Iterate — with discipline

::: notes
Say the order is not a convention: several steps are only valid in this order. This is
the map for notebook 01, which we now work through live. Tell them to open it.
:::

# Step 1 — frame it

- What is predicted, from what?
- Who would use it, for what decision?
- **Which error is worse?**

::: notes
The step everyone skips. In the tumour example the two errors are not comparable: a
false alarm costs anxiety and a follow-up test; a missed malignancy can cost a life.
Ask the room which they would rather make. That answer determines the metric - and it
must be fixed BEFORE seeing results, or it is chosen to flatter them.
:::

# Step 2 — look first

![](class_distribution.png)

::: notes
569 samples, 30 features, 357 benign against 212 malignant. Get them to compute the
consequence out loud: always answering "benign" is right 63% of the time. That is the
bar any real model has to clear, and it is why the next step exists.
:::

# Step 3 — split, before anything else

Before scaling.
Before selecting features.
Before looking at correlations with the target.

::: notes
Return to the empirical/expected risk argument - this is that slide made operational.
The moment anything from the test rows influences a decision, even computing a mean, the
test set stops being unseen. Mention stratify: it keeps class proportions identical so
the test set is a fair miniature.
:::

# Step 4 — baseline first

Predict the majority class. Nothing else.

**63%**

Everything from here is measured against that.

::: notes
Without this, "96% accuracy" has no scale. With it, they can see the model is worth
something. On the imbalanced example later, the baseline will be unbeatable on accuracy
and useless - which is the lesson, not a curiosity.
:::

# Step 5 — pipeline, not two steps

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000),
)
model.fit(X_train, y_train)
```

::: notes
Explain WHY, not just what. StandardScaler learns a mean and a standard deviation - it
is part of the model, not preparation of the data. Inside a pipeline it learns them from
the training fold every time. Done by hand, one day you scale on the full dataset and
nothing warns you. Structure beating discipline.
:::

# Step 6 — the result

- Baseline: 0.629
- Model: **0.986**

Done?

::: notes
Let "done?" hang. Someone will say no. If nobody does, ask what 0.986 does not tell
them. Then move to the confusion matrix - this is the pivot of the lesson.
:::

# What accuracy hides

![](confusion_matrix.png)

::: notes
One malignant tumour classified as benign. As a fraction it is 1 in 143 and vanishes
into the accuracy figure. As a consequence it is a person sent home who should not have
been. Say that sentence deliberately - it is the gap between a metric and what the
metric stands for, and it is the most important idea today.
:::

# Precision and recall

- **Recall** (malignant): how many malignant tumours we caught
- **Precision** (malignant): how often an alert is correct
- They trade off against each other

::: notes
Push recall to 1.0 by flagging everything and precision collapses; demand precision and
you miss cases. No single number resolves the trade-off, because the resolution is not
mathematical - it is a question about consequences and belongs to the people who bear
them. Lesson 4 builds the tools; today they just need to see the trade-off exists.
:::

# What we did not do

- No tuning
- No comparison of model families
- No check of stability across splits

And: we never looked at the test set to make a decision.

::: notes
Each omission is a later lesson - name them: 5, 6-7, 5 again. Then land the last line:
that discipline is the reason the 0.986 is worth anything at all.
:::

# Four ways a model misleads you

- Leakage
- Imbalance
- Shortcut features
- Single-split noise

All produce numbers that pass a casual review.

::: notes
Open notebook 03. Frame it: none of these is a coding error. All four run exactly as
written and return a number a reviewer, supervisor or customer would accept. This is the
section they should remember in five years.
:::

# 200 samples. 5000 random features. Coin-flip labels.

There is nothing to learn.

Anything above 50% is an artefact.

::: notes
Set it up carefully before revealing the result. Make sure they understand the data is
pure noise by construction - no signal exists. Ask what accuracy they expect.
:::

# Select features first, split second

![](leakage.png)

::: notes
77% on pure noise. Explain the mechanism: with 5000 random features some correlate with
the labels by chance, including the labels of rows that later become the test set. The
selection encoded information about the test labels. Nobody wrote a bug. Then ask what
happens with 20000 features - it gets WORSE, because there are more chances for a
spurious correlation.
:::

# The rule

Every step that **learns anything** must be fitted inside the training fold.

Scaling. Imputation. Encoding. Selection. Tuning.

::: notes
This is the practical takeaway from the whole leakage section. Pipelines enforce it
structurally. Tell them: if they remember one operational rule from today, this is it.
:::

# 99% accuracy, detecting nothing

1% positives. Answer "negative" every time.

- Accuracy: **0.986**
- Recall on positives: **0.000**

::: notes
The trained model in the notebook reaches 0.987 accuracy - barely above the trivial one
- while missing 20 of 21 positives. On a problem where the positives are the entire
reason the system exists. Accuracy is dominated by the class nobody is looking for.
:::

# Shortcut features

A column that is a **consequence** of the label, not a predictor of it.

`biopsy_scheduled` → accuracy rises → model is worthless

::: notes
Explain why it is worthless: the field is only populated after the diagnosis it is
supposed to predict. At prediction time for a new patient it is empty. Then the key
point - no metric detects this, including cross-validated ones, because the information
genuinely is in the training data. Only knowing HOW THE DATA WAS RECORDED catches it.
Ask them to imagine a hospital extract with 200 columns and no documentation.
:::

# One split is one measurement

![](split_variance.png)

::: notes
Same model, same data, 30 different seeds. Spread of 0.083 - larger than most claimed
improvements between competing methods in the literature. Pick your favourite seed and
report anything in that range; a reader cannot tell. A number quoted without
variability is incomplete. Lesson 5 makes this the centre of the course.
:::

# What the four have in common

None is a coding error.

All four run exactly as written and return a plausible number.

::: notes
They are caught by asking questions about the data and the process, not by reading the
metric. Give them the question to carry: where did this number come from, what does it
leave out, and would it still hold on data I have never touched?
:::

# A model is not objective

- It is a compressed summary of its training data, **including its injustices**
- These methods find **correlation**, not cause
- Accuracy is not the only property that matters

::: notes
Close the loop with the "when not to use ML" slide. If past decisions discriminated, a
model fitted to them discriminates - with an appearance of objectivity that makes it
harder to contest than a human decision would be. And a model can be highly accurate
and completely wrong about why, which is why it fails abruptly when conditions change.
Keep this short and serious; do not moralise.
:::

# Homework — due Friday 2 October

`Exercises/01_first_workflow.md`

Wine quality: run the workflow yourself, and **justify every decision**.

- No marks for accuracy
- Marks for methodology, and for saying what your number does not mean

::: notes
Set it explicitly, do not let them leave without knowing it exists. Walk through the
tasks briefly. Flag task 7 - repeat the split with ten seeds - as the one that connects
to what they just saw. Remind them: scaling the full dataset before splitting loses the
methodology marks regardless of the score. Also point them at the quiz and at the
project brief, so they can start thinking about a dataset.
:::

# Before next week

- Get the environment running
- Work the three notebooks in order
- Read the handout
- Take the quiz
- **Do the exercise**

Next: data — exploration, preparation, and leakage in earnest.

::: notes
Close on time. Offer to stay for setup problems. Tell them lesson 2 goes deeper into
data preparation, and that the leakage they saw today was the simple version.
:::
