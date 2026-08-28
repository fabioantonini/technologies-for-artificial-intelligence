# Where Data Comes From, and What It Costs

> **Supplementary reading — Lesson 2**
> Estimated reading time: 25 minutes
> Not examinable. It is, however, the constraint that will decide which of your
> ideas you are allowed to build.

---

## Why a computer scientist should read this

Lesson 2 treated a dataset as something that arrives: a CSV file with columns
that need cleaning. That is how it looks from inside a notebook, and it is the
one view that will never be enough professionally.

Three things this adds.

**Somebody produced every row, and it cost something.** Usually money, sometimes
labour under conditions worth knowing about, occasionally the privacy of a
person who never agreed to be in your training set.

**In the European Union, what you may do with data is law, not preference.** You
will work under the General Data Protection Regulation (GDPR). It decides
whether a project is legal before it decides whether it is good, and reading it
after the model is built is the expensive order.

**The dataset's history is part of the result.** A model trained on data
gathered under one set of assumptions carries those assumptions into every
prediction it makes.

---

## 1. The regulation you will actually work under

Regulation (EU) 2016/679 — the GDPR — has applied since 25 May 2018 across the
Union. It governs the processing of **personal data**: anything relating to an
identified or identifiable living person.

That definition is much broader than most engineers expect. A name obviously
qualifies. So does an IP address, a device identifier, a customer number, or any
combination of ordinary columns that together single somebody out.

### 1.1 Six principles, in the terms a developer meets them

Article 5 lists the principles. Four of them change how you write code.

**Purpose limitation.** Data collected for one purpose may not simply be reused
for another. The churn dataset gathered to send invoices is not automatically
available for training a marketing model. This is the principle most often
broken by accident, because a data warehouse makes everything look equally
available.

**Data minimisation.** Collect what the purpose requires, and no more. It is the
opposite instinct to the one machine learning encourages — "keep every column,
the model might find something" is a defensible engineering heuristic and an
indefensible legal position.

**Accuracy.** Inaccurate personal data must be corrected or erased. A model
trained on stale records is not merely worse; it may be unlawful.

**Storage limitation.** Keep it no longer than the purpose needs. "Forever, in
case it is useful" is not a retention policy.

The remaining two — lawfulness, fairness and transparency, and integrity and
confidentiality — are the ones lawyers spend most time on and engineers least.
(Article 5 adds a seventh duty on top of the six: **accountability**, meaning you
must be able to *demonstrate* compliance, not merely achieve it.)

### 1.2 You need a lawful basis before you start

Article 6 gives six, and you must pick one *before* processing: consent,
performance of a contract, legal obligation, vital interests, public task, or
legitimate interests.

Two practical notes. **Consent is the weakest basis**, not the strongest — it
must be freely given, specific, informed and unambiguous (Article 4(11)), and it
can be withdrawn at any time (Article 7(3)) — at which point your basis
disappears. And **legitimate interests** requires a documented
balancing test against the rights of the person, not a shrug.

### 1.3 The categories you must not touch casually

Article 9 singles out **special category data**: racial or ethnic origin,
political opinions, religious or philosophical beliefs, trade union membership,
genetic and biometric data, health, sex life and sexual orientation. Processing is
prohibited by default, with narrow exceptions.

The trap for machine learning is that you can process this data **without
collecting it**. A postcode plus a shopping history is a serviceable proxy for
ethnicity in many European cities. The regulation is concerned with effects, and
a model that infers a special category has processed one.

### 1.4 Anonymous, or merely pseudonymised?

This distinction decides whether the regulation applies at all, and it is
routinely got wrong.

**Pseudonymised** data — identifiers replaced by keys, with the mapping kept
somewhere — is *still personal data*, and the GDPR still applies in full.
Hashing an email address is pseudonymisation, not anonymisation: the hash is
stable, so it still singles a person out.

**Anonymous** data, which can no longer be attributed to a person by any means
reasonably likely to be used, falls outside the regulation entirely.

Genuine anonymisation is much harder than it looks, and the literature on
re-identification is a long series of demonstrations that it failed. Small
combinations of ordinary attributes turn out to be surprisingly unique in a
population, and a dataset released as anonymous can often be re-identified by
joining it against another one.

**The engineering lesson:** treat "we anonymised it" as a claim requiring
evidence, especially when the evidence is your own.

---

## 2. The part of the pipeline nobody puts on a slide

Supervised learning needs labels, and labels are made by people.

ImageNet — the dataset whose 2012 result opened the modern era — contains
roughly fourteen million labelled images. They were labelled through Amazon
Mechanical Turk by tens of thousands of workers across many countries, over
about two years. The compute and the architecture get the credit; the labelling
was the larger part of the effort and cost.

That pattern has not changed, it has grown. Content moderation, medical image
annotation and the human feedback behind current language models are all
performed by people, frequently in lower-income countries, frequently on
piecework rates, and frequently on material that is unpleasant or worse to look
at. Reporting on the conditions of moderation and annotation work has been
consistent enough over the past decade that a practitioner should not be
surprised by it.

Two consequences that are about engineering rather than ethics alone.

**Label quality is a function of labour conditions.** Piecework paid by the item
rewards speed, and speed produces a particular kind of noise: the easy cases are
right and the ambiguous ones are guessed. That is not random noise, and it does
not average out.

**Annotator disagreement is data, not a nuisance.** Where two competent people
disagree about a label, the task itself is ambiguous — and a model trained to a
single "gold" answer will be confidently wrong on exactly those cases. Datasets
that publish per-annotator labels are more useful than those that publish only
the majority.

---

## 3. Data cascades: why upstream problems compound

Practitioner studies of applied machine learning describe a recurring pattern
often called **data cascades**: problems introduced at collection are invisible
at the time, survive every downstream step, and surface only in deployment,
where they are most expensive to fix.

The shape is always the same. Nobody was responsible for the data itself. The
collection was done by whoever had access, the cleaning by whoever inherited it,
and the modelling by somebody who assumed both had been done properly.

Two habits are worth more than any technique in lesson 2:

- **Write down where every column came from, and when.** A column whose
  provenance nobody can state is a column nobody can defend.
- **Talk to whoever generated the data.** Fifteen minutes with the person who
  operates the sensor or fills in the form routinely explains a distribution
  that a week of exploratory analysis only describes.

---

## 4. What this means for your project

Concretely, for the final project of this course and for the first year of your
working life:

- **Prefer datasets with a documented licence and provenance.** "Found on
  Kaggle" is not provenance; the Kaggle page usually links to the actual origin.
- **Ask what the collection purpose was**, and whether your use matches it.
- **Assume the data is personal until you have established otherwise**, rather
  than the reverse.
- **Record what you did.** Not for the marker — for the version of you in
  eighteen months who has to explain the model to somebody who did not build it.

---

## Where to look next

- **Regulation (EU) 2016/679** itself. It is more readable than its reputation;
  Articles 5, 6, 9 and 22 are the ones to know, and the recitals explain the
  reasoning.
- Your national data protection authority publishes practical guidance — in
  Italy, the **Garante per la protezione dei dati personali**.
- **Datasheets for Datasets** (Gebru et al.) proposes a standard set of
  questions every dataset should answer about its own origin. Worth reading once
  and then using as a checklist.
- **Data and its (dis)contents** (Paullada et al.) surveys what dataset
  practices in machine learning research actually look like.
