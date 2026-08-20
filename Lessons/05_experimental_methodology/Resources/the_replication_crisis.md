# The Replication Crisis, and Machine Learning's Version of It

> **Supplementary reading — Lesson 5**
> Estimated reading time: 25 minutes
> Not examinable. It is the evidence that the mistakes in lesson 5 are not
> beginners' mistakes.

---

## Why a computer scientist should read this

Lesson 5 showed a signal-free dataset reporting an area under the curve of 0.93,
and a perfect classifier arriving from nothing but a lucky seed. It would be
comfortable to read those as exercises — the sort of thing that happens in a
teaching notebook and not in real work.

They happen in real work. They have happened at scale, across several fields,
for years, to careful people with reputations to protect. That is what this
document is for: not to alarm, but to establish that the discipline lesson 5
asks for is the response to a known failure rather than a matter of taste.

---

## 1. What happened outside machine learning

### 1.1 The paper that named it

In 2005 John Ioannidis published an essay with the title *Why Most Published
Research Findings Are False*. The argument is closer to lesson 4 than to
polemic: treat a published finding as a positive test result, apply the base
rate of true hypotheses in the field, and work out the probability that the
finding is real.

The answer depends on four things: how many hypotheses were plausible to begin
with, the statistical power of the study, the flexibility available in the
analysis, and how many teams were chasing the same question. In fields where the
prior probability is low and the flexibility high, the arithmetic gives a
majority of published positives being false.

That is the **base rate fallacy** from lesson 4, applied to a scientific
literature. A finding is a positive prediction; the field's prior is the base
rate; a low base rate makes precision poor however good the individual studies
are.

### 1.2 What replication attempts found

In 2015 the Open Science Collaboration attempted to replicate one hundred
studies from major psychology journals. Roughly a third produced a statistically
significant result in the same direction, and effect sizes were on average
around half the originals. Similar exercises in cancer biology and experimental
economics produced results of a comparable character.

The point is not that these fields are careless. It is that a rate like that
emerges from ordinary practice, followed by competent people, with no misconduct
required.

### 1.3 The garden of forking paths

The mechanism is the one lesson 5 measured.

Gelman and Loken's phrase describes an analysis where the researcher makes many
reasonable choices — which outliers to exclude, which covariates to include,
where to cut a continuous variable, which subgroup to examine — each one
defensible, each one made **after seeing the data**.

No single decision is dishonest. But the space of analyses that could have been
run is enormous, and the one that got published is the one that produced a
result. That is the maximum of many noisy estimates: lesson 5, section 7.3,
where a grid search over 25 combinations reported 0.80 on data containing
nothing.

The crucial and uncomfortable part: **you do not have to try all the paths for
this to bite.** It is enough that you would have tried others had the first not
worked.

---

## 2. Machine learning's own version

Our field has the same problem with different clothes, and in two respects a
worse one: our published numbers are usually a single figure with no interval,
and our benchmarks are reused by thousands of people.

### 2.1 The benchmark as a shared test set

A public benchmark is a test set used by an entire community, repeatedly, for
years. Every paper that reports on it and every architecture chosen because it
scored well is one more selection made against the same held-out data.

Collectively the field is doing exactly what lesson 5 forbids an individual from
doing: choosing on the test set. The result is that progress measured on a
long-lived benchmark is partly real and partly the accumulated optimism of
thousands of choices.

Blum and Hardt's "The Ladder" analyses this for competition leaderboards and
shows how much a leaderboard can be gamed by submissions alone — an entrant can
climb without any model at all, simply by using the feedback.

### 2.2 The comparisons that do not survive scrutiny

Reproducibility studies in several ML subfields have repeatedly found a similar
pattern: reported improvements shrink or vanish when the baseline is tuned with
the same effort as the proposed method, and when several random seeds are
reported instead of one.

The mechanism is the one you can now name. The new method was tuned hard; the
baseline was taken from a previous paper. That is not a controlled comparison,
and its result is a measurement of effort rather than of method.

### 2.3 Why our reporting makes it worse

Consider how a result is usually presented in a paper: **one number, one seed,
no interval**. Lesson 5's first notebook produced scores from 0.885 to a perfect
1.000 on one dataset with one model. A single number from that distribution,
reported without a spread, is not a wrong result — it is not a result at all.

The conference checklists that now ask for seeds, error bars and compute budgets
exist because of this.

---

## 3. What to actually do

Everything here is in lesson 5. This document only supplies the reason.

- **Hold out a test set before you begin, and touch it once.** The whole
  argument reduces to this.
- **Report a spread, not a point.** With the number of folds or seeds it came
  from.
- **Say how many things you tried.** A result chosen from forty candidates is a
  different claim from a result obtained on the first attempt, and only you know
  which one it was.
- **Tune the baseline as hard as your method.** Otherwise you have measured your
  own enthusiasm.
- **Pre-register your own analysis, informally.** Write down what you will
  measure and how you will decide, before you look. It costs ten minutes and it
  is the only defence against the garden of forking paths that survives contact
  with your own curiosity.
- **Publish the code and the seeds.** A result nobody can reproduce is not one.

---

## 4. The uncomfortable part

The literature on this is consistent about one thing: the people involved were
not cheating. They were following normal practice in their fields, making
individually reasonable decisions, under incentives that reward positive
results.

You will face the same incentives. A project that produces a clear result is
easier to present than one that produces an honest uncertainty; a model that
beats the baseline makes a better report than one that does not.

Lesson 5's discipline costs you the score you would otherwise have reported.
That is not a side effect of doing it properly — it *is* doing it properly, and
the difference between 0.93 and 0.50 on data with no signal is the size of what
is at stake.

---

## Where to look next

- **Ioannidis, "Why Most Published Research Findings Are False" (2005)** —
  short, and the arithmetic is accessible with lesson 4 behind you.
- **Open Science Collaboration, "Estimating the Reproducibility of Psychological
  Science" (2015)** — the large replication attempt, and its methodology.
- **Gelman and Loken, "The Garden of Forking Paths" (2013)** — why no
  p-hacking is required for the effect to appear.
- **Blum and Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning
  Competitions" (2015)** — leaderboard overfitting, with a proposed remedy.
- **Pineau et al., on the NeurIPS reproducibility programme** — what a field
  actually did about it, and what it cost.
