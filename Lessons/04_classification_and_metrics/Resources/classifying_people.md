# Classifying People

> **Supplementary reading — Lesson 4**
> Estimated reading time: 30 minutes
> Not examinable. The impossibility result in Section 3 is mathematics, though,
> and it is the kind you should not meet for the first time in a meeting.

---

## Why a computer scientist should read this

Lesson 4 built a confusion matrix for disk drives. Replace the drives with
people — loan applicants, defendants, patients, job candidates — and every cell
of that matrix becomes a decision about somebody's life.

Nothing in the mathematics changes. What changes is that the four cells are no
longer equivalent, that different groups of people may land in them at different
rates, and that "make the model more accurate" stops being a sufficient answer.

This document is about a specific, well-documented case, and about a result that
came out of the argument over it. The result is the important part: it says that
several reasonable definitions of a fair classifier are **mathematically
incompatible**, so the choice between them is a choice somebody has to make and
defend.

---

## 1. The case

In 2016 the newsroom ProPublica examined COMPAS, a commercial risk-assessment
tool used in parts of the United States to score defendants on the likelihood of
reoffending. Scores of this kind inform decisions about bail and sentencing.

ProPublica reported that among defendants who did **not** go on to reoffend,
Black defendants were roughly twice as likely as white defendants to have been
labelled high risk. In lesson 4's vocabulary: the **false positive rate**
differed by race.

Northpointe, the company behind the tool, responded that the scores were
**calibrated**: among defendants given a particular score, the proportion who
actually reoffended was about the same regardless of race. A score of 7 meant
the same thing for everybody.

Read those two claims again. They are both about the same predictions, both
supported by the same data, and **both true**.

That is not a paradox to be resolved by looking more carefully. It is a
mathematical fact, and it was proved shortly afterwards.

---

## 2. Three definitions of a fair classifier

Each is reasonable. Each can be written in terms of the confusion matrix you
already know, computed separately for each group.

**Calibration within groups.** Among everyone the model gives score $s$, the
same fraction actually turn out positive, whatever group they belong to. This is
what you want if the score is going to be read as a probability: a 7 means a 7.

**Equal false positive rates.** Among people who are *not* positive, the same
fraction get flagged, whatever their group. This is what you want if being
wrongly flagged carries a cost — an unnecessary detention, a refused loan, an
invasive follow-up.

**Equal false negative rates.** Among people who *are* positive, the same
fraction get missed. The mirror image, and what you want when a miss is the
expensive error.

The last two together are usually called **equalised odds**.

---

## 3. The impossibility result

Here is the part worth knowing precisely.

> If two groups have **different base rates** — different underlying proportions
> of positives — then no classifier can be calibrated within groups *and* have
> equal false positive rates *and* equal false negative rates, except in
> degenerate cases: a perfect classifier, or one with no predictive power at
> all.

This was established independently by Kleinberg, Mullainathan and Raghavan
(2016) and by Chouldechova (2017), and it is not an empirical finding about
COMPAS. It is a theorem about confusion matrices.

### 3.1 Why it happens

The intuition needs only lesson 4's arithmetic.

Calibration ties the *precision* of each score to the truth. Equalised odds ties
the *rates* at which the two kinds of error occur. Precision and the error rates
are connected through the base rate — as lesson 4's section 8.3 showed, the same
model at the same threshold has different precision when the positive rate
changes.

So if the base rates differ and you fix the precision to be equal, the error
rates must differ. Fix the error rates instead, and the precision must differ.
The base rate is the hinge, and you cannot pin down both ends of it at once.

### 3.2 What follows, and what does not

**What does not follow:** that fairness is hopeless, or that the choice is
arbitrary, or that since you cannot have everything you may as well ignore all
of it.

**What does follow:** that "is this model fair?" is not a well-posed question
until somebody says which definition they mean. And that the choice between them
is a decision about **which error you would rather make**, which is precisely
the decision lesson 4 made with money in section 7.2 — here with something that
cannot be priced in euros.

That reframing is the useful part. You already know how to choose a threshold
from costs. This is the same act, with the costs contested and the parties
disagreeing about them, and the mathematics is not going to settle it for you.

---

## 4. The base rate itself is not neutral

One more turn of the screw, and it is the part most often skipped.

The impossibility result takes the base rates as given. But a base rate is
measured, and it is measured by a process.

If the label is "was arrested again" rather than "committed another offence",
then the base rate reflects policing patterns as well as behaviour. A model
trained on it predicts the measurement, faithfully, including whatever produced
the measurement.

This is the classification version of the lesson-2 point about provenance: the
model learns the data-generating process, and the data-generating process
includes the humans in it. No amount of care about the confusion matrix reaches
a problem that lives in the label definition.

**The question to ask before any of the fairness arithmetic:** what exactly does
the positive class mean, who decided, and how was it recorded?

---

## 5. What this looks like in practice

You are unlikely to be handed a decision about criminal justice. You are
reasonably likely, within a few years, to be handed one about who gets a loan,
whose insurance claim is reviewed, whose CV is read by a person, or which
patients are flagged for follow-up.

Four things that transfer directly from lesson 4:

- **Report the confusion matrix per group**, not just overall. An aggregate
  metric can hide two very different experiences, in exactly the way accuracy
  hid the failing drives.
- **Say which fairness definition you used** and why, in the same breath as the
  number. Section 3 is the reason the qualifier is not optional.
- **Check whether the base rates differ** between groups before promising
  anything. If they do, you are inside the impossibility result whether you
  invoke it or not.
- **Ask where the label came from.** Section 4.

And one that does not come from the mathematics at all: the people affected by
these systems are rarely in the room when the threshold is chosen. Being the
person who says so is part of the job.

---

## Where to look next

- **Angwin, Larson, Mattu and Kirchner, "Machine Bias" (ProPublica, 2016)** —
  the original piece, and the methodology supplement published with it.
- **Chouldechova, "Fair Prediction with Disparate Impact" (2017)** — the
  impossibility result, stated compactly.
- **Kleinberg, Mullainathan and Raghavan, "Inherent Trade-Offs in the Fair
  Determination of Risk Scores" (2016)** — the same result, arrived at
  independently.
- **Barocas, Hardt and Narayanan, *Fairness and Machine Learning*** — freely
  available online, and the standard textbook treatment. Chapter 2 covers
  everything above properly.
- The **EU AI Act** classifies several of the applications named in Section 5 as
  high risk, with obligations attached. Worth knowing it exists before somebody
  asks you about it.
