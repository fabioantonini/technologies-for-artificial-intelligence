# Where Linear Models Still Win

> **Supplementary reading — Lesson 3**
> Estimated reading time: 25 minutes
> Not examinable. It explains why the simplest model in the course is also the
> one you are most likely to meet in production.

---

## Why a computer scientist should read this

Lesson 3 presented linear regression as the first method: exact solution,
iterative solution, readable coefficients. It would be easy to file it under
"the thing we did before the interesting models".

That would be a mistake about the industry you are about to enter. Linear and
logistic models remain in production, in some of the highest-stakes decisions
made about people, decades after more accurate methods became available. The
reasons are worth understanding, because they are not about accuracy — and
because the day you propose replacing one, these are the objections you will
meet.

The second half of this document is about **regression to the mean**, which is a
statistical fact you have already met and a reasoning trap you have almost
certainly fallen into.

---

## 1. Credit scoring: interpretability as a legal requirement

Consumer credit scoring is a large, mature, heavily regulated application, and
much of it is logistic regression on a few dozen carefully engineered features —
a "scorecard".

That is not technological conservatism. Three forces hold it there.

**Somebody has to be told why.** A refused applicant is entitled to an
explanation, and "the model said so" is not one. A scorecard produces a sentence:
*your application scored below the threshold, mainly because of the number of
recent credit enquiries.* A gradient-boosted ensemble can be probed with
post-hoc tools, but the explanation is then an approximation of the model rather
than the model itself — which is a materially weaker position in front of a
regulator.

**The model must be auditable years later.** Scorecards are inspected,
challenged and defended. A model whose behaviour can be read off a table of
coefficients can be argued about; one that cannot has to be re-run.

**Monitoring is easier.** When a coefficient means "per standard deviation of
this feature", a shift in the input distribution has a readable consequence. The
model tells you how it will misbehave before it does.

The accuracy gap is real but usually small — on tabular data with well
engineered features, the difference between a good linear model and a good
ensemble is often a couple of points. Whether those points are worth losing the
three properties above is a business decision, and it frequently goes the other
way.

## 2. Clinical risk scores: models a human must be able to apply

Medicine is full of scores that a doctor can compute at a bedside: a handful of
inputs, small integer weights, a total that maps to a risk band.

These are almost always fitted as logistic regressions and then **deliberately
degraded** — coefficients rounded to integers, continuous variables cut into
bands. The rounding costs accuracy on purpose, in exchange for a model that can
be applied without a computer, remembered, taught, and checked by a second
clinician.

The engineering lesson generalises well beyond medicine: **the deployed model is
the one that gets used**, and usability is part of performance. A model that
requires infrastructure the setting does not have has an effective accuracy of
zero there.

## 3. Insurance pricing: the model has to be filed

In much of Europe, insurance rating factors are filed with a regulator and must
be justified — and some factors are prohibited outright. Generalised linear
models dominate, partly because the structure makes the filing possible: each
factor's effect is a number that can be stated, defended, and constrained.

Note the shape of the constraint. It is not "the model must be accurate". It is
"the model must be *expressible*, and each part of it must be defensible on its
own". Flexible models are poor at that by construction: their power comes from
interactions that resist being stated separately.

## 4. Econometrics: when the coefficient *is* the result

In economics and the social sciences, a regression is often run not to predict
anything at all but to estimate one number — the effect of a policy, a price, a
treatment — with an interval around it.

This is a different activity wearing similar clothes, and confusing the two
causes real errors in both directions. A model with excellent predictive
accuracy can have coefficients that mean nothing causally; a model with modest
predictive accuracy can estimate one causal effect well. Lesson 3's warning
about collinearity is the practical face of this: two columns carrying one fact
predict fine together and neither coefficient can be interpreted alone.

---

## 5. Regression to the mean, and why the world seems to punish praise

The word "regression" comes from a phenomenon, not from the algorithm. Francis
Galton, studying inherited height in the 1880s, observed that the children of
unusually tall parents were on average taller than the population but *shorter
than their parents* — "regression towards mediocrity", in his phrase.

The mechanism is not biological. It is arithmetic, and it applies to anything
measured with noise.

**The picture.** Any measurement is signal plus noise. An extreme measurement is
extreme partly because the signal was high and partly because the noise happened
to help. The signal persists on remeasurement; the luck does not. So the second
measurement is, on average, less extreme.

That is all. And it produces one of the most robust illusions in human
reasoning.

### 5.1 The illusion, in its classic form

Kahneman describes being told by flight instructors that praising a cadet for a
good manoeuvre was followed by a worse one, while criticising a bad manoeuvre
was followed by an improvement — and that they had therefore learned to
criticise rather than praise.

Both observations were correct. The conclusion was wrong. An unusually good
manoeuvre is followed by a worse one whatever the instructor says, and an
unusually bad one by a better one, because performance varies and extremes are
partly luck.

The instructors had run an uncontrolled experiment on themselves and drawn the
opposite of the truth from it. **Reality had rewarded them for punishment and
punished them for praise**, and nothing in the data would have told them
otherwise.

### 5.2 Where you will meet it

- **Sports.** The "cover jinx" — an athlete featured after an exceptional run
  performs worse afterwards. Yes, and they would have anyway.
- **Medicine.** Patients enrol in a trial when their symptoms are at their
  worst. Many improve regardless of treatment, which is why an untreated control
  group exists.
- **Business.** The worst-performing region gets an intervention and improves.
  The intervention gets the credit.
- **Machine learning.** A feature selected because it scored highest in this
  sample will, on new data, score lower. Lesson 5 gives this its own name and
  measures it: the maximum of many noisy estimates is biased upward.

That last one is the connection worth carrying. Regression to the mean and the
optimism of model selection are the same arithmetic seen from two directions,
and recognising it in a story about flight instructors makes it much harder to
miss in a leaderboard.

---

## Where to look next

- **Galton, "Regression Towards Mediocrity in Hereditary Stature" (1886)** —
  short, readable, and the origin of the word.
- **Kahneman, *Thinking, Fast and Slow***, the chapters on regression to the
  mean. The flight-instructor story is there in his own account.
- **Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes
  Decisions and Use Interpretable Models Instead" (2019)** — argues the position
  of this document considerably more forcefully than it does.
- **Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning***,
  chapter 3, for the statistical treatment of everything lesson 3 derived.
