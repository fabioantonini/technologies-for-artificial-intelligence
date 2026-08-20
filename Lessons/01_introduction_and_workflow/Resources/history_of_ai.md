# A History of Artificial Intelligence

> **Supplementary reading — Lesson 1**
> Estimated reading time: 30 minutes
> Not examinable in itself. The *pattern* it describes is.

---

## Why a computer scientist should read this

Not for the anecdotes. For three things you can use.

**The pattern repeats.** Roughly every twenty years the field has produced a genuine
advance, followed by claims well beyond the evidence, followed by a correction painful
enough to cost a generation of funding. You are living through another iteration.
Recognising the shape is a professional skill.

**Most "new" ideas are old.** Backpropagation dates from the 1970s and 80s.
Convolutional networks from 1989. Attention from 2014. What changed was data and
compute, not principle. Knowing this stops you from mistaking a scaling result for a
conceptual one.

**The failures are more instructive than the successes.** Each winter had a specific
technical cause, and each cause is something you will meet in this course under a
different name.

---

## 1. Foundations (1943–1956)

![](hist_1943_neuron.png)

*1943: the first artificial neuron — inputs, weights, a threshold.*

![](hist_1950_imitation.png)

*1950: Turing's imitation game, which replaced "can machines think" with a question that can actually be settled.*

![](hist_1956_dartmouth.png)

*1956: Dartmouth, where the field acquired both its name and its habit of optimistic timelines.*

### 1943 — A logical calculus

Warren McCulloch, a neurophysiologist, and Walter Pitts, a logician, publish *A Logical
Calculus of the Ideas Immanent in Nervous Activity*. They model a neuron as a binary
threshold unit: it outputs 1 when the weighted sum of its inputs reaches a threshold,
0 otherwise.

$$y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

They show networks of such units can compute any logical proposition. The claim is
philosophical as much as technical: thought might be computation.

The weights are set by hand. Nothing learns yet. But the unit in Lesson 9 is this unit,
with a smoother threshold.

### 1949 — Hebbian learning

Donald Hebb proposes that connections strengthen when neurons fire together — "cells
that fire together wire together". The first mechanism for *learning* as weight
modification, and the ancestor of every update rule in this course.

### 1950 — The imitation game

Turing opens *Computing Machinery and Intelligence* by declaring "Can machines think?"
too meaningless to deserve discussion, and replaces it with an operational test: can an
interrogator distinguish a machine from a person by conversation alone?

Note what he did, because this course does it repeatedly. He replaced a question that
cannot be settled with a **measurable proxy**, then argued the proxy was good enough.
Every metric you choose is the same move, and carries the same obligation: stay aware
of the gap between what you measure and what you mean.

The paper also answers nine objections, including the one that a machine can only do
what it is told. His reply — that this assumes we can foresee the consequences of what
we tell it — has aged well.

### 1956 — Dartmouth

John McCarthy, Marvin Minsky, Claude Shannon and Nathaniel Rochester convene a summer
workshop at Dartmouth College. The proposal coins **artificial intelligence** and
states that a significant advance can be made if a carefully selected group of
scientists works on it together for a summer.

Ten people, two months, for a problem unsolved seventy years later. The field's
characteristic optimism is present at its naming.

---

## 2. The symbolic era and the first winter (1956–1980)

![](hist_1957_perceptron.png)

*1957: the perceptron — the first machine that improved with experience rather than with rewriting.*

![](hist_1969_xor.png)

*1969: XOR, a function a single-layer perceptron cannot represent, and the book that ended the first wave of funding.*

### 1957 — The perceptron

Frank Rosenblatt builds a machine that **learns its own weights** from examples,
adjusting them whenever it makes a mistake. The Mark I Perceptron is hardware:
photocells, potentiometers, motors.

Rosenblatt proves the *perceptron convergence theorem* — if the data can be separated
by a straight line, the procedure will find one in finite time. A real guarantee, with
a condition attached.

The press ignores the condition. The New York Times reports the Navy expects a machine
that will walk, talk, see, write and reproduce itself.

### 1966 — ELIZA

Joseph Weizenbaum writes a program that reflects a user's statements back as
questions, imitating a Rogerian therapist. It has no understanding whatsoever — the
whole program is pattern substitution.

People confide in it. Weizenbaum's secretary asks him to leave the room. He spends much
of his later career arguing against the interpretations his program invited, which
makes ELIZA the first case study in a problem that never went away: **fluency is read
as understanding**, and the gap between the two is invisible to the user.

### 1969 — Perceptrons

Minsky and Papert publish a rigorous analysis showing a single-layer perceptron cannot
represent XOR — a function no straight line separates.

The mathematics is correct. The reception is not: the limitation applies to *one
layer*, and multi-layer networks were known to be more expressive. What was missing was
a way to train them. The distinction is lost in the retelling.

**First AI winter.** Funding collapses through the 1970s. Neural network research
becomes close to unpublishable for a decade.

You will meet XOR again in Lesson 9, where a hidden layer disposes of it in four lines.

### 1970s — Expert systems

The symbolic approach delivers its most practical results: MYCIN diagnoses bacterial
infections at the level of specialists; DENDRAL infers molecular structure. Both encode
hand-written rules elicited from human experts.

They work, within narrow domains, and expose the approach's limit. The rules are
expensive to acquire, brittle at the edges, and do not generalise. This is precisely the
difficulty that motivates learning from data — the argument in Section 2 of the
handout, discovered the hard way.

---

## 3. Connectionism returns, then falters (1986–1995)

![](hist_1986_backprop.png)

*1986: backpropagation — the chain rule applied to a network, which made several layers trainable at last.*

### 1986 — Backpropagation

Rumelhart, Hinton and Williams popularise an efficient way to compute the gradient of
the loss with respect to every weight in a multi-layer network, making such networks
trainable. The technique had been derived several times before — Linnainmaa in 1970,
Werbos in 1974 — without taking hold.

Minsky and Papert's objection is answered. Enthusiasm returns.

Lesson 9 derives this algorithm.

### 1989 — Convolutional networks

Yann LeCun applies backpropagation to handwritten digit recognition with an
architecture that exploits the spatial structure of images: local receptive fields,
shared weights, pooling. LeNet reads postal codes commercially.

Every idea in Lesson 10 is here, in 1989. What was missing was data and compute.

### Early 1990s — The second winter

Networks are hard to train, need more data than is available, and take too long on the
hardware of the era. Expert system companies collapse as their products prove
unmaintainable. Funding contracts again.

---

## 4. The statistical era (1995–2012)

![](hist_1995_svm.png)

*1995 onwards: support vector machines and the statistical turn, where theory caught up with practice for about fifteen years.*

The field retreats from "simulating intelligence" to "estimating functions from data".
Ambition down, rigour sharply up.

**Support vector machines** (Cortes and Vapnik, 1995) come with theory that explains
*why* they generalise, plus the kernel trick for non-linear boundaries. Lesson 6.

**Ensembles**: Random Forests (Breiman, 2001) and boosting (Freund and Schapire, 1997)
combine weak models into strong ones, and win competitions for a decade. Lesson 7.

**Statistical learning theory** matures — the bias-variance decomposition,
generalisation bounds, cross-validation as standard practice. Lesson 5.

This period produced most of what actually gets deployed, and most of this course. It
is unglamorous and it is where the reliable tools come from.

---

## 5. Deep learning and scale (2012–)

![](hist_2012_imagenet.png)

*2012: AlexNet. Conceptually little was new — what had changed was data and compute.*

### 2012 — AlexNet

Krizhevsky, Sutskever and Hinton win ImageNet with a convolutional network, cutting the
error rate from 26% to 15% — an unprecedented margin.

The architecture is recognisably LeNet, larger. What changed:

- **Data**: ImageNet, 1.2 million labelled images, assembled over years
- **Compute**: two consumer GPUs
- **Details that mattered**: ReLU activations, dropout, data augmentation

A conceptual advance would have been visible in 1989. This was a scaling result, and
saying so is not a criticism — it is the most important fact about the last decade.

### 2017 onwards

The transformer architecture (Vaswani et al., 2017) removes the sequential bottleneck
in recurrent networks, enabling training at unprecedented scale. Combined with
self-supervised objectives — predict the hidden token, the technique from notebook 02 —
it produces systems trained on internet-scale corpora without annotation.

The capabilities surprised most researchers. The accompanying claims would have been
familiar to anyone reading coverage of the perceptron in 1958. Both observations are
true at once, and holding them together is the point.

---

## 6. What the pattern tells you

| Period | Advance | Overclaim | Correction |
|---|---|---|---|
| 1957–1969 | Perceptron learns from data | Machines that walk and talk | XOR; first winter |
| 1986–1995 | Backprop makes depth trainable | Neural networks solve everything | Too little data and compute; second winter |
| 2012– | Scale delivers real capability | ? | ? |

Three things to carry forward.

**Distinguish a demonstrated result from an extrapolated promise.** Rosenblatt's
convergence theorem was true. "It will reproduce itself" was not. Both appeared in the
same press cycle, and telling them apart required reading the condition on the theorem.

**Ask what changed.** When a result improves sharply, ask whether the idea is new or
the resources are. Both are valuable; they license very different predictions about
what comes next.

**The winters were caused by the gap between claims and evidence**, not by the
technology being worthless. The perceptron was genuinely useful; it was oversold. The
professional habit that prevents this is the same one this course teaches for
evaluating a model: state precisely what was measured, on what data, and what that
does not establish.

---

## Primary sources

| Year | Work | Why it is worth the hour |
|---|---|---|
| 1943 | McCulloch & Pitts, *A Logical Calculus of the Ideas Immanent in Nervous Activity* | The first artificial neuron, in its authors' terms |
| 1950 | Turing, *Computing Machinery and Intelligence* | Readable, funny, and the objections still land |
| 1958 | Rosenblatt, *The Perceptron: A Probabilistic Model...* | The learning rule and the theorem, with its condition |
| 1969 | Minsky & Papert, *Perceptrons* | The XOR argument as actually written, rather than as remembered |
| 1986 | Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* | Two pages that ended the first winter |
| 1998 | LeCun et al., *Gradient-Based Learning Applied to Document Recognition* | Lesson 10, fourteen years early |
| 2012 | Krizhevsky et al., *ImageNet Classification with Deep CNNs* | The paper that restarted the field |
