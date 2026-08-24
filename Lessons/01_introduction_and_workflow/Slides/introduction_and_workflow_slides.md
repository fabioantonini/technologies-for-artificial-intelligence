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
Introduce yourself briefly, then establish the two facts that shape everything: this is
a fundamentals course, and it assumes programming ability but no ML background.

Ask for a show of hands: who has trained a model before? Usually a handful. Tell that
group they have an advantage of about two lessons, and a disadvantage that lasts
longer, because they will have habits to unlearn around evaluation - most self-taught
practice picks up the modelling and skips the validation.

Practical notes to give now: the repository is public to them via git, everything runs
offline in Docker, and there is no cost attached to any part of this course. Nobody
needs an application programming interface (API) key, a cloud account or a
graphics processing unit (GPU).
:::

# What we will cover

| | |
|---|---|
| 1–2 | Workflow, data preparation |
| 3–4 | Regression, classification |
| **5** | **Experimental methodology** |
| 6–7 | k-NN, support vector machines, trees, ensembles |
| 8 | Unsupervised learning |
| 9–10 | Neural networks, CNNs |

::: notes
Point at lesson 5 and say it is deliberately in the middle, not at the end. It is the
lesson about whether results mean anything, and everything after it depends on it.
Most courses put this material last, where it arrives too late to change how students
work.

Say explicitly that there are no LLMs in this course - they belong to another module in
the programme. Some will be disappointed. Tell them the material here is what those
systems are built on: the transformer is a neural network, trained by gradient descent
on a loss function, and evaluated - badly, often - with the same statistical tools we
will spend ten weeks on.

If asked "will we do deep learning?": yes, lessons 9 and 10, but on a laptop CPU and
with the emphasis on understanding rather than scale.
:::

# What this course is not

- Not a tour of current AI products
- Not a library tutorial
- Not a course where you call `.fit()` and report the number

::: notes
Be direct here. They can learn the scikit-learn API from documentation in an afternoon,
and they will. What takes a semester is knowing which number to trust.

The concrete promise to make: if they leave this course able to look at someone else's
reported result - a paper, a colleague's notebook, a vendor's benchmark - and identify
what is wrong with the evaluation, the course succeeded. That skill outlasts every
library we touch.
:::

# Your advantage, and your gap

**Advantage:** you can program. You will implement methods, not only call them.

**Gap:** you can produce results faster than you can judge them.

::: notes
This is the framing for the whole course, so spend a real minute on it. Handout section
1 develops it.

Within three weeks they will write thirty lines that give 95% accuracy on something.
The hard question is whether it means anything. Say plainly: the most common failure in
this field is not a bug - the code runs exactly as written - it is a number that was
never valid in the first place.

Contrast with the audience this material used to serve: PhD engineers who had the
mathematics and lacked the coding. This group is the mirror image, so the emphasis
moves from "make it run" to "make it true".
:::

# How you will be assessed

- **Weekly exercises** — set every Friday, due the next
- **Final project** — an end-to-end study, with peer review
- **Written exam** — closed book, drawn from the handouts

::: notes
Set expectations now. The exercises are not optional and they build towards the
project: by lesson 5 they should be able to write their project's evaluation plan, by
lesson 7 to have a draft comparison.

Point them at Assessment/ in the repository. The project brief is available today, so
they can start thinking about a dataset - the topic must be confirmed by lesson 4.

Emphasise the principle that runs through all three components: methodology outweighs
accuracy. Two students with different scores can both earn full marks. A spectacular
score obtained by leaking the test set will not - the rubric caps methodology marks at
40% for that, and today's third notebook shows exactly why.

The exam is closed book and drawn from the handouts, including the derivations. Say
this now so nobody treats the handouts as optional reading.
:::

# The three things you get each lesson

- **Handout** — the reference text, with the mathematics
- **Slides** — what we do in the room
- **Notebooks** — runnable implementations

Plus a quiz, and the exercise.

::: notes
Explain the division of labour, because it is unusual and it matters.

Derivations are NOT on the slides. They are in the handout, to be read afterwards. In
the room we do results, intuition and code. This is what lets the lecture move at a
sensible pace while the course stays rigorous - and it is why the exam can ask them to
reproduce a derivation.

Slides and notebooks arrive on the Friday; the handout follows by the Monday, since it
is study material rather than lecture support.

The quizzes are self-check and ungraded, but three or four questions in each require
reasoning about a derivation rather than recall - those are the closest thing to the
exam they will see before the sample papers.
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
Do this live and take ten minutes over it. Making sure everyone has JupyterLab running
today is far cheaper than fixing it during lesson 3, when they also have an exercise
due.

Explain the two-part design: the Docker image carries the environment, the repository
carries the content. New lessons arrive with `git pull` - a few megabytes, seconds -
not a new multi-gigabyte image. The image only changes if the libraries change, and
they will be told when that happens.

Common problems: port 8888 already in use (edit docker-compose.yml to 8889), and the
first pull taking several minutes on the room's wifi. Warn them the first start is
slow and every later one is not.
:::

# Today

- What learning from data means
- A short history, and what it teaches
- The three kinds of learning
- The end-to-end workflow
- Four ways a model misleads you

::: notes
Agenda slide. Flag the last item as the one that matters most today: everything before
it is orientation, and that section is where the real subject of the course appears.

Timing: roughly 50 minutes to the break, then the three kinds of learning, then the
workflow worked live, then the failure modes. Homework is set in the last five minutes
- do not let it slip, it is due next Friday.
:::

# A problem you cannot specify

Write a program that decides whether an email is spam.

- Filter certain words? Spam adapts.
- Filter certain senders? They change.
- The exceptions grow until nobody can maintain it

::: notes
Work this conversationally. Ask the room for rules and write two or three on the board,
then break each one: "buy now" - what about a legitimate shop newsletter? Sender
blacklists - what about a compromised account belonging to a colleague?

Get them to feel the frustration before offering the alternative. The point to land, and
it is handout section 2: the program is not hard to write, the RULE is impossible to
state. You recognise spam instantly and cannot articulate how.

If someone proposes machine learning immediately, slow them down - the interesting part
is understanding precisely which difficulty it addresses.

Take each row in turn and let someone in the room supply the exception before you reveal it. The pattern to draw out: every rule is defensible and every rule breaks, and the list has no end.
:::

# Every rule buys an exception

![](spam_rules.png)

::: notes
Read the four rows down the left, and the exception each one buys on the
right. Take them one at a time - the effect depends on the accumulation.

The point is not that these are bad rules. Each is a reasonable idea, and
each fails on a case that is obviously not spam to any human reader. Ask
the room for a fifth rule and then for what breaks it; somebody will
supply both within a few seconds, which is the demonstration.

The line at the bottom is the one to leave them with: you cannot state the
rule, but you recognise the answer. That gap is the whole reason this
course exists, and every method in it is a way of closing it from
examples instead of from specification.
:::

# The inversion

You cannot state the rule.
But you can **recognise** the answer.

![](rules_versus_learning.png)

::: notes
This is the conceptual move that defines the field, so say it slowly while the figure is
up.

Traditional programming: you supply rules and data, the machine produces answers.
Machine learning: you supply data and answers, and the machine produces the rules. The
arrows literally reverse.

Everything else in this course is machinery for doing that search well - and for
telling whether the rules it found are real or an artefact of the sample you happened
to collect.

Worth adding: this framing is due to Arthur Samuel in the 1950s, and it is still the
cleanest one-sentence description of what changed.
:::

# Making it precise

- Inputs from a space `X`, outputs from a space `Y`
- Pairs (x, y) drawn from an unknown distribution `D`
- A loss `L(ŷ, y)`: the cost of answering ŷ when the truth is y
- Goal: find f: X → Y that predicts well

::: notes
Keep this brisk - handout section 2.1 has the full treatment and they will read it.

Two things to stress about D. It is UNKNOWN, which is why we need data at all. And it is
assumed FIXED - the same distribution generates the training data and the data we will
meet later. That assumption is quiet and it fails constantly in practice: the hospital
changes its equipment, the users change, the season turns. When it fails, models degrade
without any warning from their metrics. Promise them we return to it.

The loss function is where the domain enters the mathematics. Choosing it is a decision
about which mistakes matter, and we make that choice explicitly in twenty minutes.

Point at the dashed box. Three of these objects we handle directly - the inputs, the function we search for, the predictions. The fourth, the distribution the data comes from, we never see. Everything difficult in this course comes from that one box being out of reach.
:::

# The setup, in one picture

![](learning_setup.png)

::: notes
Everything on the previous slide, drawn once so it can be pointed at for
the rest of the course.

Trace the path with a finger: a pair is drawn from D, x goes into f, f
produces y-hat, the loss compares y-hat against the y that came with it.
Then say what is unknown - D is unknown, and it is the only thing we
actually care about.

Worth naming now, because it returns in lesson 5: we never see D. We see a
sample from it, and every claim we make about a model is an inference from
the sample to the distribution. Most of the methodology later in the
course is about not fooling yourself in that step.
:::

# What we actually want

The average loss over all data the world might produce — including data that does not
exist yet:

$$R(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ L(f(x), y) \right]$$

::: notes
Name it: expected risk, sometimes called true risk or generalisation error.

Then deliver the punchline and pause on it - we cannot compute this. D is unknown, the
expectation is over data that has not been collected, and no amount of cleverness gets
around it.

This is the central obstacle of the field, and everything else in the course follows
from it. If a student takes one idea away today, this is a good candidate.
:::

# What we can actually compute

Average loss over the finite sample we happen to hold:

$$\hat{R}_S(f) = \frac{1}{m} \sum_{i=1}^{m} L(f(x_i), y_i)$$

::: notes
Empirical risk. Minimising it is empirical risk minimisation, and essentially every
method in this course is an instance of it - linear regression, logistic regression,
trees, neural networks. Different function classes, same principle.

Ask the room: what could go wrong with minimising this instead of the thing we actually
want? Give them a moment. Someone usually says "it might not generalise", which is
exactly right and is the next slide.
:::

# The gap that explains everything

- We **minimise** the empirical risk
- We **care about** the expected risk

![](risk_gap.png)

::: notes
The central slide of the lesson. Walk the figure carefully.

Empirical risk falls monotonically as the model gets more flexible - more parameters
always fit the sample better, and with enough of them you reach zero by storing every
example and reciting its answer.

Expected risk turns back up. The shaded region between the curves is overfitting: the
model has learnt the sample rather than the pattern.

The consequence for practice: performance on data you fitted to is ALWAYS optimistic and
sometimes meaningless. Handout section 2.2 makes the argument in full.

Note the dotted line - the best trade-off - and say we spend lesson 5 learning to find
it honestly, and lessons 3 and 7 meeting the two standard tools for controlling it,
regularisation and ensembling.
:::

# The same idea, on real data

![](overfitting.png)

::: notes
Three fits to identical points. Left: a straight line cannot follow the curve, so it is
wrong everywhere in the same way - high bias, poor on training AND test data. Right:
degree-18 passes through nearly every point and oscillates wildly between them - it
will be badly wrong on anything new.

Ask which one they would choose without seeing the dashed line. That is the real
situation: the true pattern is never visible, only the sample.

Point out the trap in the right-hand panel - it has the LOWEST training error of the
three. If you selected a model by training error you would pick it every time. That is
precisely why we hold data out.
:::

# Which gives us the one rule for today

Hold out part of the data.
Never let the fitting procedure touch it.
Measure there.

::: notes
Say clearly: this is not bureaucratic caution, and give the reason rather than the
instruction. Handout sections 2.3 to 2.5 carry the full argument.

The difficulty is not that a sample is small. It is that if the same data both CHOOSES
the model and JUDGES it, the judgement stops being an unbiased estimate. The analogy
that works: a revision session that happens to rehearse exactly the exam questions.
Nobody cheated - the same questions both shaped the preparation and judged it.

Why a held-out set repairs it: if the test set played no part in choosing f, it stays
independent of f, and the error measured on it is an unbiased estimate of the expected
risk - the quantity we just said was not computable. Worth putting on the board.

Stress what that argument depends on: the INDEPENDENCE of the test set, not its size
or its proportion. The moment those rows influence any decision - a mean for scaling,
a ranking for feature selection, a comparison between models - the independence breaks
and the estimate is optimistic by an unknown amount. That is why the rule is "nothing
is learned before the split" rather than "keep some data aside".

Warn them we will break exactly that independence later today and watch 77% accuracy
appear out of coin-flip labels.

If someone asks "how much do we hold out?" - typically 20-30%, and the trade-off is
real in both directions: a bigger test set gives a more stable estimate (the standard
error falls as one over root n), a bigger training set gives a better model. Give them
the concrete number from notebook 01: 143 test examples at 0.986 accuracy carries a
standard error of about one percentage point, so the third decimal place is noise.
With a few hundred examples both sides hurt at once, which is what cross-validation
exists for - lesson 5.

If someone asks "what if I try ten models and report the best?" - they have just
selected using the test set, and the number is optimistic again. That is the bridge to
the validation set, and to lesson 5. Today we need only train and test because we make
no choices: one model, no tuning.
:::

# When *not* to use machine learning

- The rule is **known** — write it
- The data is **not representative** of deployment
- Errors are **catastrophic and unexplainable**
- The data encodes an **injustice** you would automate
- A **simple baseline** already suffices

::: notes
Unfashionable and immediately useful - handout section 3.

First point: nobody should train a classifier to detect whether a number is even. It
sounds obvious and is violated constantly, usually because ML is the interesting option
and a rule is boring.

Second: a model learns the distribution it was trained on. Sample from one hospital,
deploy in another, and the measured accuracy says very little.

Third: learned models are wrong sometimes, in ways that are hard to predict and often
hard to explain. If a mistake is unrecoverable and nobody reviews the output, the
question is not whether accuracy is high enough but whether anyone can tell when it has
failed.

Fourth: a model is not neutral because it is mathematical. It is a compressed summary of
the decisions in its training data. We return to this at the end of the lesson.

Fifth: often a threshold on one variable solves 90% of the problem, is understandable by
everyone, and needs no maintenance. Establishing baselines - step 4 of the workflow - is
partly how you discover this before building something complicated.
:::

# Seventy years in one picture

![](ai_timeline.png)

::: notes
Open the history section with the whole arc visible, then walk the individual moments.
Resources/history_of_ai.md has the full version with primary sources - tell them it is
worth an hour and is not examinable in itself.

The two grey bands are the AI winters. Point at them now and say the question for the
next ten minutes is what caused them, because the answer is not "the technology did not
work".
:::

# 1943 — the first artificial neuron

A neuron as a threshold unit: it fires when the weighted sum passes θ.

![](hist_1943_neuron.png)

::: notes
A neurophysiologist and a logician. Their claim was philosophical as much as technical:
thought might be computation. They showed networks of such units can compute any
logical proposition.

Weights were set by hand - nothing learned yet. That comes in 1957.

Tell them the unit in lesson 9 is this unit with a smoother threshold. The history is
not decoration; it is the same objects under different names.

Walk the diagram: inputs, a weight on each, a sum, and a hard threshold. Then point at the right-hand panel - the step function is what makes it all-or-nothing, and it is exactly what gets replaced by a smooth sigmoid in lesson 4 and by ReLU in lesson 9. Same object, softened, so that it becomes differentiable and therefore trainable.
:::

# 1950 — Turing's imitation game

"Can machines think?" — replaced by something observable.

![](hist_1950_imitation.png)

::: notes
Highlight the MOVE rather than the test. Turing opens by declaring the original question
too meaningless to deserve discussion, and substitutes an operational one.

That is exactly what they do every time they choose a metric: replace something that
cannot be measured - "is this model good?" - with something that can. It carries the
same obligation, to stay aware of the gap between the proxy and the thing itself. The
whole of lesson 4 is about that gap for classification.

Recommend the paper. It is readable, funny, and answers nine objections including "a
machine can only do what it is told" - his reply, that this assumes we can foresee the
consequences of what we tell it, has aged extremely well.

The screen is the whole design: it removes appearance, voice and everything except the behaviour being tested. Ask the room what this test does NOT measure - understanding, consciousness, correctness - and note that Turing knew, and argued the substitution was worth making anyway.
:::

# 1956 — Dartmouth

The term *artificial intelligence* is coined.

![](hist_1956_dartmouth.png)

::: notes
Let the room react to "two months". Seventy years later the problem is open.

McCarthy, Minsky, Shannon and Rochester convened it. The optimism is not incidental -
it is present at the field's naming and it recurs on a cycle, which is the argument of
this whole section.

These are the seven topics from the actual proposal. Read two or three aloud - 'self-improvement', 'randomness and creativity' - and let the room judge how much of the list is settled. Then note that the proposal's authors expected substantial progress on all of it in one summer.
:::

# 1957 — the perceptron learns

Rosenblatt: weights adjusted **from examples**, not set by hand.

![](hist_1957_perceptron.png)

::: notes
The first machine that learns. Physical hardware - photocells, potentiometers, motors.

Stress the CONDITION on the theorem: "if linearly separable". The theorem is true and
that clause is carrying enormous weight, as 1969 will show.

Then the contrast: the New York Times reported the Navy expected a machine that would
walk, talk, see, write and reproduce itself. Same year, same machine. Ask them which of
those two statements a careful reader could have checked.

The faded lines are earlier states of the boundary. Every time the machine misclassified an example it nudged the weights, and the line moved. That is the whole idea: the model is corrected by its own mistakes, which is still what gradient descent does in lesson 3 - just with a smoother update rule.
:::

# 1969 — the first winter

A single-layer perceptron cannot represent XOR.

![](hist_1969_xor.png)

::: notes
Draw XOR on the board - four points, alternating labels - and let them see that no
single straight line separates them. Takes thirty seconds and lands permanently.

Then the key point: the result applies to ONE LAYER. Multi-layer networks were already
known to be more expressive; what was missing was a way to train them. That distinction
was lost in the retelling, funding collapsed, and neural network research became close
to unpublishable for a decade.

In lesson 9 a hidden layer disposes of XOR in four lines of Keras. Fifteen years of lost
time for a limitation that a correct reading would have scoped much more narrowly.

The transferable lesson: a correct result about a restricted case was received as a
verdict on the whole approach. That failure mode is alive and well in how ML results
get reported today.

Use the figure instead of the board. AND on the left: one line, done. XOR on the right: the dashed lines are attempts, and each leaves a point on the wrong side. Ask them to try to find one that works before you move on - the impossibility is more convincing when they have looked for it themselves.
:::

# 1986 — backpropagation

An efficient way to train multi-layer networks.

![](hist_1986_backprop.png)

::: notes
Rumelhart, Hinton and Williams. Note the technique had been derived before - Linnainmaa
1970, Werbos 1974 - without taking hold. Ideas need context as much as correctness,
which is a useful thing for a researcher to know.

The second winter arrived in the early 1990s: networks were hard to train, needed more
data than existed, and took too long on the hardware. Expert system companies collapsed
as their products proved unmaintainable.

Lesson 9 derives this algorithm.

Two passes: forward to compute a prediction, backward to distribute the error across every weight, including the hidden ones. That second arrow is what was missing in 1969 - not expressiveness, but a way to assign blame to a unit that never sees the target directly. Lesson 9 derives it.
:::

# 1995–2010 — statistics takes over

From "simulating intelligence" to "estimating functions from data".

![](hist_1995_svm.png)

::: notes
Say that most of this course lives here, and it is where the reliable tools come from.
SVMs arrived with theory explaining WHY they generalise. Random forests and boosting won
competitions for a decade. The bias-variance decomposition and cross-validation became
standard practice.

It is unglamorous and it is what actually gets deployed. Lessons 5, 6 and 7 are this
period, and lesson 5 in particular is its central contribution.

The dotted line also separates the classes perfectly, and would score identically on this training data. The SVM picks the solid one - the boundary furthest from both classes. Ask which they would trust on a new point near the middle. That intuition is what the theory of the period made precise, and it is lesson 6.
:::

# 2012 — AlexNet

The architecture was recognisably LeNet, from 1989.

![](hist_2012_imagenet.png)

::: notes
The most important slide in the history section.

What changed was DATA - ImageNet, 1.2 million labelled images, assembled over years -
and COMPUTE, two consumer GPUs. Plus details that mattered: ReLU activations, dropout,
augmentation.

A conceptual advance would have been visible in 1989, when LeCun was already reading
postal codes with convolutional networks. This was a scaling result.

Saying so is not a criticism. It is the single most important fact about the last decade,
and it licenses very different predictions than a conceptual breakthrough would.

Read the bars left to right. Two ordinary years, then the drop, then everyone adopts the approach and the error keeps falling past human performance by 2015. Point out that the years after 2012 fell further than 2012 itself did - the story is not one heroic result, it is a field switching methods.
:::

# The pattern

| Advance | Overclaim | Correction |
|---|---|---|
| Perceptron learns | Machines that walk and talk | XOR, first winter |
| Backprop trains depth | Networks solve everything | No data, second winter |
| Scale delivers | ? | ? |

::: notes
Leave the question marks and ask the room what belongs there. It usually produces a good
discussion, and there is no answer you need to supply.

Three things to draw out. First, distinguishing a demonstrated result from an
extrapolated promise: Rosenblatt's theorem was true, "it will reproduce itself" was not,
and both appeared in the same press cycle. Second, asking what changed - idea or
resources. Third, and this is the one they will use professionally: the winters were
caused by the gap between claims and evidence, not by the technology being worthless.
The perceptron was genuinely useful; it was oversold.

Then make the connection explicit: the habit that prevents this is the same one we use
to evaluate a model - state precisely what was measured, on what data, and what that
does not establish.
:::

# Three kinds of learning

The taxonomy divides by **where the target comes from** — not by algorithm.

::: notes
Correct the common misconception immediately, before it takes hold. It is not about
which algorithm you use: the same estimator can serve more than one setting, and they
will see exactly that in notebook 02 where the supervised regression and the
self-supervised task run identical code.

Handout section 5 has the full taxonomy.
:::

# Supervised

Each example carries a label a person produced.

- Category → **classification**
- Continuous → **regression**
- You can check answers against truth

::: notes
Lessons 3, 4, 6 and 7 are all supervised, so this is most of the course.

Ask what the expensive part is. Someone will say the labels, and that is right: in
practice labels - not algorithms, not compute - are what limits most real projects.
Somebody has to produce them, correctly, in quantity, and for many problems that is
expensive or simply impossible.

Give the concrete example: ImageNet took years of human annotation, and it is the reason
2012 happened when it did rather than in 1995.
:::

# Unsupervised

No targets at all. Find structure.

- Groups → clustering
- Fewer dimensions → dimensionality reduction
- Odd points → anomaly detection

::: notes
Lesson 8.

Plant the difficulty now: there is no accuracy to report. The algorithm ALWAYS returns
clusters, whether or not the data contains any groups. Whether they mean something is a
judgement about the domain and cannot be delegated to a metric.

This is genuinely harder than supervised learning to do well, which is the opposite of
how it is usually taught - it looks easier because there is no labelling effort.
:::

# Self-supervised

Hide part of the input. Predict it from the rest.

Nobody annotates anything — and the supervision is real.

![](self_supervision.png)

::: notes
The one they will have heard about without a clean definition, so define it precisely.

The reconstruction task is a PRETEXT: nobody wants a flavanoid predictor, which is what
notebook 02 builds. The point is that solving it forces the model to represent how the
parts of an input relate, and that representation transfers to tasks you do care about.

Scale the idea up - hide a word in a sentence, hide a patch of an image - and it is how
modern large models are trained. Free supervision is precisely why they scale: text and
images exist in enormous quantities and nobody has to annotate them.

This is the honest one-paragraph answer to "how does ChatGPT learn?", and it is worth
giving them, since they will be asked it at every family dinner.

Walk it left to right: one column hidden, the rest used to predict it. Nobody labelled anything, yet there is a genuine target. Then scale the idea in their heads - hide a word in a sentence, a patch in an image - and they have the training principle behind every large model they have heard of.
:::

# One dataset, three questions

![](kinds_of_learning.png)

::: notes
Work notebook 02 here, live. The same 178 wines appear in all three panels.

Left: what the algorithm sees when there are no labels - just points in space. Middle:
labels supplied by a person. Right: groups found by geometry alone.

Point out that the cluster colours do not match the label colours. k-means numbers its
groups arbitrarily, and without labels there is nothing to align to. That arbitrariness
IS the point of the unsupervised setting.

Warn them the strong agreement here - adjusted Rand index 0.897 - is a happy accident of
this dataset, where the chemical groups really are geometrically separated. Do not
expect it. In most real data, clustering finds structure that corresponds to nothing you
care about.
:::

# Where the target comes from

| | Target | Cost | Measurable? |
|---|---|---|---|
| Supervised | A person | Expensive | Yes |
| Unsupervised | None | Free | Not directly |
| Self-supervised | The input | Free | Only the pretext |

::: notes
Read the cost column aloud: it explains why self-supervision dominates wherever data is
abundant and annotation is not, and why supervised learning still wins whenever someone
has already paid for good labels.

Then read the right-hand column and flag the trap: in two cases out of three you cannot
straightforwardly measure success at the thing you actually want. That is a much bigger
practical problem than the choice of algorithm, and it is why unsupervised results are so
often oversold.
:::

# The workflow

![](ml_workflow.png)

::: notes
Say the order is not a convention: several steps are only valid in this order, and step
3 is the one that cannot move - hence the colour.

This is the map for notebook 01, which we now work through live. Tell them to open it
and follow along rather than watch.

Handout section 6 has each step in full. The two they will be tempted to skip are 1 and
4 - framing and baseline - and those are the two that make the difference between a
result and a number.
:::

# Step 1 — frame it

- What is predicted, from what?
- Who would use it, for what decision?
- **Which error is worse?**

::: notes
The step everyone skips, and the one the exercise marks hardest.

Our example: predicting whether a breast tumour is malignant from measurements of cell
nuclei. A model here would be a screening aid, never a diagnosis - say that clearly, it
matters.

The two errors are not comparable. A false alarm costs anxiety and a follow-up test; a
missed malignancy can cost a life. Ask the room which they would rather make, and let
them answer before showing any metric.

That answer determines the metric, and it must be fixed BEFORE seeing results. A metric
chosen afterwards is a metric chosen to flatter - with enough candidates, one always
looks good.

Two of these cells are fine and two are not, and the two failures are not the same size. Accuracy adds them up as if they were. Ask which cell they would accept more of - and note that answering requires knowing who bears the cost, which is not a question about data.
:::

# Which error is worse?

![](error_costs.png)

::: notes
Two problems, the same confusion matrix, and opposite answers about which
cell hurts.

Ask the room before reading either side. Missing a malignant tumour versus
raising a false alarm; blocking a legitimate email versus letting spam
through. Nobody needs statistics to rank these, and that is the point -
the ranking comes from the application, and it has to arrive before any
modelling decision is made.

Lesson 4 turns this into a threshold that can be computed. Today it is
enough that they see the question is not a modelling question, and that
answering it with accuracy answers it by accident.
:::

# Step 2 — look first

![](class_distribution.png)

::: notes
569 samples, 30 features, 357 benign against 212 malignant.

Get them to compute the consequence out loud: always answering "benign" is right 62.7%
of the time. That number is the bar any real model has to clear, and it is why step 4
exists.

Mild imbalance here. Later today we will see what happens at 99 to 1, where the same
reasoning destroys accuracy as a metric entirely.
:::

# Step 3 — split, before anything else

Before scaling.
Before selecting features.
Before looking at correlations with the target.

![](split_scheme.png)

::: notes
Return to the risk_gap figure verbally - this slide is that argument made operational.

The moment anything from the test rows influences a decision, even something as innocent
as computing a mean for scaling, the test set stops being unseen and the number it gives
you is optimistic.

Mention stratify=y: it keeps the class proportions identical in both parts, so the test
set is a fair miniature of the whole. Without it, on a small or imbalanced dataset, you
can get a test set whose composition differs from the training set by chance.

They will violate this deliberately in the third notebook and see what it costs.

The caption under the test set is the whole discipline: touched once, at the very end. Say that every additional look spends some of its independence - if you test twenty variants and report the best, you have selected on the test set and the number is no longer honest.
:::

# Step 4 — baseline first

Predict the majority class. Nothing else.

**62.7%**

Everything from here is measured against that.

::: notes
Without this, "96% accuracy" has no scale at all. With it, they can see whether the
model is worth anything.

DummyClassifier does this in two lines, and the exercise requires it.

Foreshadow: on the imbalanced example later, the baseline will be unbeatable on accuracy
and completely useless in practice. That is the lesson, not a curiosity - it is how you
discover that accuracy is the wrong metric for your problem.
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
Explain WHY, not just what.

StandardScaler LEARNS something - a mean and a standard deviation per feature. That
makes it part of the model, not preparation of the data. Inside a pipeline it learns
them from the training fold every time the pipeline is fitted, including inside each
cross-validation fold in lesson 5.

Write the two steps separately and one day you will scale using statistics computed over
the whole dataset. Nothing warns you: no error, no exception, just a better number than
you deserve.

This is structure beating discipline, and it is the single most useful habit in the
scikit-learn API.

The top row is the mistake, and note where 'split' sits: after the scaling. The mean used to scale the training rows was computed with the test rows included. Nothing errors, nothing warns, and the score comes out slightly too good - which is the worst kind of bug, because there is nothing to debug.
:::

# What the pipeline prevents

![](pipeline_versus_manual.png)

::: notes
The same two steps, done in the two orders, with what each one shows the
test set.

On the manual side the scaler is fitted on everything, so the mean and the
standard deviation it subtracts already contain the test rows. The model
is then evaluated on data it has, in a small way, already seen. The score
comes out slightly too high, and nothing anywhere raises an error.

Emphasise that last part. This is not a mistake that announces itself: it
produces a working notebook and a plausible number. Lesson 2 measures what
it costs on a real dataset, and lesson 5 gives it its name.
:::

# Step 6 — the result

- Baseline: 0.629
- Model: **0.986**

Done?

::: notes
Let "done?" hang for a few seconds. Someone will say no.

If nobody does, ask what 0.986 does not tell them. Then move to the confusion matrix -
this is the pivot of the entire lesson.
:::

# What accuracy hides

![](confusion_matrix.png)

::: notes
One malignant tumour classified as benign.

As a fraction it is 1 in 143 and vanishes into the accuracy figure. As a consequence it
is a person sent home who should not have been.

Say that sentence deliberately and let it sit. It is the gap between a metric and what
the metric stands for, and it is the most important idea in today's lesson.

Then the practical form: accuracy collapses every kind of mistake into one number, and
our two mistakes are not comparable. The confusion matrix is the minimum tool for seeing
them separately.
:::

# Precision and recall

- **Recall** (malignant): how many malignant tumours we caught
- **Precision** (malignant): how often an alert is correct
- They trade off against each other

::: notes
Use the figure to make the trade-off physical: the threshold is a line you slide, and
sliding it moves errors from one column to the other.

Push recall towards 1.0 by flagging anything suspicious, and precision collapses - the
clinic drowns in unnecessary biopsies. Demand high precision and you start missing
cases.

No single number resolves this, because the resolution is not mathematical. It is a
question about consequences and it belongs to the people who bear them - here, patients
and clinicians, not the person who trained the model.

Lesson 4 builds the tools properly: receiver operating characteristic
(ROC) curves, precision-recall curves, choosing an
operating point. Today they need to see that the trade-off exists and that it is not
the model's decision to make.
:::

# The trade-off, drawn

![](precision_recall_tradeoff.png)

::: notes
One model, one dataset, and the threshold swept from one end to the other.

Follow the two curves in opposite directions. Lower the threshold and
recall rises - we catch more of what is there - while precision falls,
because more of what we flag is wrong. There is no setting where both are
at their best, and no amount of better modelling removes the trade.

So the question of whether a model is good has no answer until somebody
says which error is worse, which is the slide from earlier arriving again
with a number attached. Lesson 4 spends an hour here.
:::

# What we did not do

- No tuning
- No comparison of model families
- No check of stability across splits

And: we never looked at the test set to make a decision.

::: notes
Name where each omission gets addressed: tuning and stability in lesson 5, model
families in 6 and 7.

Then land the last line, which is the point of the slide: that discipline is the reason
the 0.986 is worth anything at all. Every one of the four failures we are about to see
comes from breaking it in some form.
:::

# Four ways a model misleads you

- Leakage
- Imbalance
- Shortcut features
- Single-split noise

All produce numbers that pass a casual review.

::: notes
Open notebook 03. Frame it before starting: none of these is a coding error. All four
run exactly as written and return a number that a reviewer, a supervisor or a customer
would accept.

This is the section they should still remember in five years. Handout section 7
summarises it, and every one of the four has been found in published work and in
shipped products.
:::

# 200 samples. 5000 random features. Coin-flip labels.

There is nothing to learn.

Anything above 50% is an artefact.

::: notes
Set this up carefully before revealing anything. Make sure they understand the data is
pure noise BY CONSTRUCTION - the labels come from a random number generator and no
feature carries any information about them.

Ask what accuracy they expect. The honest answer is 50%, chance.
:::

# Select features first, split second

![](leakage.png)

::: notes
77% on pure noise, against 50% for the honest pipeline.

Explain the mechanism: with 5000 random features, some correlate with the labels by
chance. Selecting the best 20 using the whole dataset picks exactly those - including
the ones that correlate with the labels of rows that later become the test set. The
selection encoded information about the test labels, and the model inherited it.

Nobody wrote a bug. Selection feels harmless because it fits no model - but it LEARNS
from data, so it belongs inside the training fold.

Then ask what happens with 20000 features instead of 5000. The illusion gets STRONGER,
because there are more opportunities for a spurious correlation. That is the "try this"
at the end of the notebook.
:::

# The rule

Every step that **learns anything** must be fitted inside the training fold.

Scaling. Imputation. Encoding. Selection. Tuning.

::: notes
The practical takeaway from the whole leakage section, and the thing to put on the board
if anything goes on the board today.

Pipelines enforce it structurally, which is why we used one from the first notebook.

Tell them the exercise checks this: scaling the full dataset before splitting caps the
methodology marks regardless of the result obtained.
:::

# 99% accuracy, detecting nothing

1% positives. Answer "negative" every time.

- Accuracy: **0.986**
- Recall on positives: **0.000**

::: notes
The trained model in the notebook reaches 0.987 accuracy - barely above the trivial one
- while missing 20 of the 21 positives in the test set.

On a problem where the positives are the entire reason the system exists.

Accuracy is dominated by the class nobody is looking for. Ask them what metric they
would report to someone deploying this, and steer towards recall on the positive class -
then note that recall alone is also gameable, by flagging everything, which is why
lesson 4 needs a whole session.

Read the bottom-left cell in both matrices: 21 missed, then 20 missed. The accuracies differ by one thousandth. If you reported only accuracy, these two systems look identical - and one of them is a constant function that never looks at the input.
:::

# Two models, one accuracy

![](imbalance_matrix.png)

::: notes
Two confusion matrices side by side. Give them a moment to spot that the
accuracies agree to three decimal places.

Left is the model that answers negative every time and has learned nothing
whatever. Right is a trained model. 0.986 against 0.987 - and the one on
the right finds a single positive out of twenty-one, which is barely
better than nothing but is not nothing.

Ask what a report containing only the accuracy would say about these two,
and let the silence do the work. Then point at the bottom-left cell of
each: that is the number the application cares about, and it is the one
accuracy averages away.
:::

# Shortcut features

A column that is a **consequence** of the label, not a predictor of it.

`biopsy_scheduled` → accuracy rises → model is worthless

::: notes
Explain why it is worthless: biopsies are scheduled once malignancy is already
suspected, so the field is populated AFTER the diagnosis it is supposed to predict. At
prediction time for a new patient it is empty.

Then the point that makes this different from the others: no metric detects it,
including cross-validated ones, because the information genuinely IS in the training
data. The model is not cheating; the data is.

Only knowing how the data was recorded catches it. Ask them to imagine a hospital
extract with 200 columns and no documentation - which is the normal case - and to think
about how they would find this.

The question to carry: at the moment a prediction is needed, does this value exist yet?

The shaded region is the trap. Everything to the right of the blue marker happens after the moment a prediction is needed, so none of it can be an input. Give them the question in the title as the thing to ask of every column in every dataset they are handed.
:::

# Does this value exist yet?

![](shortcut_timeline.png)

::: notes
A timeline, with the moment of prediction marked, and everything that
happens after it shaded.

Walk left to right: the scan is taken, measurements are recorded - both
before the prediction is needed, both usable. Then the shaded region: the
diagnosis is made, the biopsy is scheduled. A model leaning on
biopsy_scheduled scores beautifully in the notebook and has nothing to
read at the moment it would actually be used.

The question in the title is the one to make a habit of, and it is worth
saying that it cannot be answered by looking at the data. It is answered
by asking somebody who knows how the data is produced.
:::

# One split is one measurement

![](split_variance.png)

::: notes
Same model, same data, 30 different random seeds. Accuracy ranges from 0.917 to 1.000 -
a spread of 0.083.

That is larger than most claimed improvements between competing methods in published
work. Let that land.

Pick your favourite seed and you can report anything in that range, and a reader seeing
one number cannot tell which you got. This is not fraud; it is the default behaviour of
anyone who runs the code once.

Cross-validation replaces the single draw with an average and - more usefully - reports
the spread. On this data, 5-fold gives 0.960 ± 0.030.

A result quoted with no indication of variability is incomplete. Lesson 5 makes this the
centre of the course.
:::

# What the four have in common

None is a coding error.

All four run exactly as written and return a plausible number.

::: notes
They are caught by asking questions about the data and the process, not by reading the
metric.

Give them the three questions to carry out of this room: where did this number come
from, what does it leave out, and would it still hold on data I have never touched?

Then the professional version, connecting back to the history section: the habit of
distinguishing what was demonstrated from what was extrapolated is the same habit,
applied to your own work instead of someone else's.
:::

# A model is not objective

- It is a compressed summary of its training data, **including its injustices**
- These methods find **correlation**, not cause
- Accuracy is not the only property that matters

::: notes
Close the loop with the "when not to use ML" slide. Handout section 8.

If past decisions discriminated, a model fitted to them discriminates too - and it does
so with an appearance of objectivity that makes it HARDER to contest than a human
decision would be. "The algorithm decided" is a much harder claim to argue with than
"the manager decided".

Second point: nothing in empirical risk minimisation distinguishes a cause from a
coincidence that predicts well in this sample. A model can be highly accurate and
completely wrong about why - which is exactly why it fails abruptly when conditions
change.

Third: whether a decision can be explained to the person affected, whether errors are
recoverable, and who bears the cost of being wrong are engineering requirements. They
belong in step 1, where they can still change what you build, not in a paragraph at the
end of a report.

Keep this short and serious. Do not moralise - state it and move on.
:::

# Homework — due Friday 2 October

`Exercises/01_first_workflow.md`

Wine quality: run the workflow yourself, and **justify every decision**.

- No marks for accuracy
- Marks for methodology, and for saying what your number does not mean

::: notes
Set it explicitly - do not let anyone leave without knowing it exists and when it is due.

Walk the tasks briefly. Task 1 is framing, and there is no single right answer: quality
is an integer from 3 to 9, so they must decide how to treat it and defend the choice.
Task 7 - repeat the split with ten seeds - connects directly to the figure they just saw.

Remind them: scaling the full dataset before splitting loses the methodology marks
regardless of the score. And there are no marks for accuracy anywhere in this course.

Also point them at the quiz - 20 questions, ungraded, and the ones marked "reasoning" are
the closest thing to the exam - and at the project brief, so they can start thinking
about a dataset. Topic confirmed by lesson 4.
:::

# Before next week

- Get the environment running
- Work the three notebooks in order
- Read the handout
- Take the quiz
- **Do the exercise**

Next: data — exploration, preparation, and leakage in earnest.

::: notes
Close on time and offer to stay for setup problems - it is worth twenty minutes now to
avoid three people stuck for a fortnight.

Tell them lesson 2 goes deeper into data preparation, and that the leakage they saw
today was the simple, visible version. The next one is subtler.

If anyone asks what to read beyond the handout: Resources/history_of_ai.md for context,
and the scikit-learn "common pitfalls" page, which is the library's own account of
everything in notebook 03.
:::
