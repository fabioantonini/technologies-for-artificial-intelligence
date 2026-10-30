---
title: "The Energy Cost of Training"
subtitle: "Supplementary reading, lesson 9 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "4 December 2026 · reading time about 25 minutes"
---

> Not examinable. The habit in Section 6 below — refusing to repeat a carbon figure without knowing which assumptions were stacked to produce it — is not mathematics, but it is what separates a professional from someone quoting a headline.

---

## Why a computer scientist should read this

Section 3.4 of the handout made the network wider and watched what happened.
On the acceptance test, the mean of five runs went from **0.6048** with one
hidden unit to **0.9445** with eight, within three points of the rig's ceiling
of 0.97, and to 0.9549 with sixteen. On the drift problem, two hidden units
*can* solve it — Section 3.2 writes such a network down by hand — yet training
found that solution in only 6 runs of 20, while eight units found it in 20 of
20. The handout drew the obvious conclusion: a layer wider than the problem
strictly needs gives training more chances to start near a good set of lines,
"which is why networks used in practice are wider than their task requires."

That conclusion is correct, and at the scale the field now works at it is also
an instruction to spend electricity. Making the network bigger is the cheapest
lever in Lesson 9 — cheap because a laptop trains a network on 2,250 sensors in
seconds. This document is about what the same lever costs when it is pulled
industrially, and what a professional is now expected to measure and report.

Three more hooks, all physical rather than statistical. Section 4.4 of the
handout noted that one backward sweep costs about as much arithmetic as one
forward pass, so a gradient for every weight costs roughly one more prediction.
Arithmetic, and the memory traffic that feeds it, are the two things a chip
turns into heat, so Section 4.4 is a claim about watts dressed as a claim about
complexity. Section 6.2 says the same thing outright about early stopping: the
remedies could not be ranked on accuracy, but "what early stopping certainly
bought is time, 39 epochs instead of 200" — and time on a chip is energy. And
the drift table of Section 3.4 is a lesson in reliability as much as in
accuracy: with two hidden units most runs stopped at three clouds of four, with
eight none did. Every restart a more reliable configuration makes unnecessary
is a run that never draws power. **Reliability is an energy strategy.**

---

## 1. A price list, starting from this lesson

Section 3.4 of the handout trained twenty networks at each of three widths on
the drift problem: sixty networks, for one table of three rows. It then trained
five at each of six widths on the acceptance test, and Sections 5.5 and 6.2 ran
three per row for two more tables of three rows. That is over a hundred trained
networks behind four small tables — the normal ratio. **The
published number is the surviving fraction of what was run**, so any energy
accounting that counts only the final run understates a project by one or two
orders of magnitude. All of it still finishes in minutes on a laptop central
processing unit (CPU), which is exactly why the cost never comes up in class.

One step up. Dodge and colleagues, measuring on a commercial cloud in 2022,
trained three sizes of DenseNet — a convolutional image classifier — on the
handwritten-digit benchmark MNIST (Modified National Institute of Standards and
Technology). Each job took 20 to 25 minutes on one
graphics processing unit (GPU) and drew **20 to 38 watt-hours**: a 60-watt lamp
for half an hour. The same table records five image classifiers trained on
ImageNet, from 1.7 kilowatt-hours (kWh) for the smallest to **237.6 kWh** for the
largest, which held four GPUs for nine days. A factor of 140 within one model
family, and about ten thousand between the digit classifier and the top of that
range.

Now the historical anchor, which Lesson 1's supplementary reading introduced.
AlexNet was trained for about 90 passes over 1.2 million images, and its authors
report **five to six days on two GTX 580 GPUs**, cards rated at 244 watts each.
Two of them near full load for five and a half days is about 64 kWh — a rough
ceiling on GPU energy alone, excluding host machine, disks and cooling. At the
United States average of 0.429 kilograms of carbon dioxide equivalent (CO2e) per
kWh that Patterson and colleagues use, that is roughly 28 kg CO2e, or by the
Environmental Protection Agency's (EPA) average passenger-vehicle rate **about a
hundred kilometres of driving**. The most consequential training run in the
modern history of the field cost about one commute. That is the baseline against
which every alarming figure since should be read.

What changed was not the cost of one run but their number and size. An OpenAI
analysis in 2018 estimated that compute in the largest published training runs
grew more than **300,000 times between AlexNet in 2012 and AlphaGo Zero in
2017** — doubling every 3.4 months against Moore's Law's two years. Hardware did
not become 300,000 times more efficient in six years, so most of that ratio is
real energy. Neither endpoint is a language model: the growth is a property of
the field, not of one application of it.

---

## 2. Where the energy actually goes

A chip spends energy on three things: arithmetic, moving numbers to and from
where the arithmetic happens, and the overhead of being programmable at all. The
canonical measurements are Mark Horowitz's 2014 plenary at the International
Solid-State Circuits Conference (ISSCC), for a 45-nanometre process at 0.9 volts. A
32-bit floating-point addition costs about **0.9 picojoules (pJ)**, a
multiplication about **3.7 pJ**, an on-chip cache read about **20 pJ**. An access
to off-chip dynamic random-access memory (DRAM) costs **1 to 2 nanojoules** — in
Horowitz's words, "a couple of orders-of-magnitude higher than the cost of an
internal cache access or functional operation." One instruction on a
general-purpose processor carries about **70 pJ** of overhead, against "a few pJ
for an operation."

Three consequences follow, and each explains hardware you have used. **Moving a
number can cost a thousand times more than computing with it**, which is why
fast numerical code is a discipline of data locality, and why matrix
multiplication is the friendliest operation in computing: $O(n^3)$ arithmetic on
$O(n^2)$ data, so the movement amortises. Forward propagation, in the matrix
form of the handout's Section 3.3 — the examples stacked as the rows of $X$,
multiplied by $W^{[1]}$ — is exactly this. **Programmability is expensive**, so a
processor issuing one instruction per operation spends most of its energy on
bookkeeping; widening to many data lanes amortises it, which is what a GPU does,
and casting the multiply-accumulate array into silicon amortises it further,
which is what a tensor processing unit (TPU) does. **Fewer bits means less
movement**, which is most of why training in 16-bit formats became standard.

Put the network of the handout's Section 6.1 on this price list. Two hidden
layers of 512 units on 8×8 digits is **301,066 weights and biases**. A forward
pass costs about two floating-point operations per parameter per example.
Section 4.4 says a backward sweep costs about as much as a forward pass; counted
operation by operation it is up to about twice as much, and taking that upper
end makes a full step roughly 58 megaflop for a mini-batch of 32 — the size
notebook 03 uses — or about **0.13 millijoules of arithmetic** at Horowitz's
4.6 pJ per multiply-add. But those weights occupy 1.2 megabytes, will not sit in
a small cache, and reading them once from DRAM costs about **0.23 millijoules** —
and they are read at least twice per step. Arithmetic and movement are the same
order of magnitude, and movement is the side that grows. This is an illustration
on 2014-era figures, not a measurement; absolute energies are far lower today,
but the *ratio* has widened rather than narrowed.

There is a memory side to the same coin, which the lighter lesson leaves out
and the advanced version in `Advanced/09_neural_networks_advanced/` treats:
the backward sweep needs every layer's weighted sums $Z^{[l]}$ and activations
$A^{[l]}$ from the forward pass, so they must be **stored** between the two.
For the network above that is 256 kilobytes at a mini-batch of 32: nothing.
Scale the layers to an image model and it is gigabytes, which is why
practitioners recompute activations rather than store them, trading arithmetic
back for memory, and why batch size is set by the accelerator rather than by
anything statistical. Section 4.1 of the handout chose mini-batches of 32 for a
statistical reason — much cheaper than using all the data, and still pointing
roughly the right way on average — and at scale that choice carries a hardware
clause: the batch size is often not yours to choose.

---

## 3. The datacentre multiplier, and what it hides

Electricity delivered to a server is not electricity drawn from the grid.
Cooling, power conversion, lighting and distribution losses consume more. The
standard metric is **power usage effectiveness (PUE)**, defined by the industry
consortium The Green Grid in 2007 and standardised as ISO/IEC 30134-2:2016 by
the International Organization for Standardization and the International
Electrotechnical Commission. It is a ratio: total facility energy over energy
delivered to the information technology equipment. A PUE of 1.5 means half as
much again is spent on everything that is not computing.

The Uptime Institute's annual survey puts the global average near **1.56**,
essentially flat for five consecutive years — half the surveyed facilities are
over eleven years old, and retrofitting is expensive. Patterson and colleagues
cite the United States national average as 1.58 in 2018 and 1.59 in 2020, and
report **1.11** for the Google datacentre where their experiments ran. The same
computation on identical hardware therefore reaches the meter having cost about
40% more in a typical enterprise facility than in a purpose-built one.

Here is the part students reliably get wrong, and the wrong answer is reasonable.
PUE looks like an efficiency metric, so a lower PUE ought to mean less energy per
unit of work. It does not, because it is a **ratio with the computing in the
denominator**: replace your servers with ones twice as wasteful and your PUE
improves, since cooling is now a smaller fraction of a larger total. PUE says
nothing about utilisation — a facility idling at 5% load can post an excellent
figure — nothing about the carbon intensity of the electricity, nothing about
water, and nothing about emissions embodied in manufacturing the hardware. It
measures one overhead well and is silent on the rest. The companion metrics that
cover part of the gap — water usage effectiveness (WUE), carbon usage
effectiveness (CUE), energy reuse factor (ERF) — are far less reported, which is
why the European regulation in Section 8 below names them explicitly.

At sector level the record is genuinely surprising. Masanet and colleagues, in
*Science* in 2020, found global datacentre electricity roughly flat from 2010 to
2018 — around 1% of world electricity — while the computing work done there grew
about **550%**. Efficiency gains, consolidation and the shift from server rooms
to hyperscale facilities absorbed nearly the whole increase, which is the
strongest available argument against naive extrapolation. It is not a law,
though: the International Energy Agency (IEA) put datacentre consumption at
**415 terawatt-hours in 2024, some 1.5% of world electricity**, and projected
roughly a doubling by 2030. The gains that absorbed the 2010s were largely
one-time structural shifts, and they cannot be repeated.

---

## 4. The same run, a different grid

Energy is not carbon. A kilowatt-hour's emissions depend entirely on what
generated it, and the spread between grids exceeds almost any efficiency gain
available in software. Ember's 2025 review put the **global average at 473 grams
of CO2e per kWh in 2024** and the European Union (EU) average at 213. Within
Europe, Sweden's power sector was around 11 grams and France's around 41, both
dominated by non-fossil generation; Poland's was the Union's highest, in the
region of 650 to 700, and India's national figure is around 670. Cleanest to
dirtiest here is **more than fifty to one**.

Dodge and colleagues measured what that means for an identical job. The same
workload run in sixteen regions of one commercial cloud provider emitted from
about **7 kilograms of CO2e in the best region to about 26 in the worst** — same
code, same hardware, same duration, differing only in which grid the building is
attached to. Their month-by-month lines are nearly flat, so this is a property of
the region rather than the season. Patterson and colleagues report the same
effect from inside an operator: the carbon-free fraction of supply varies roughly
five to ten times "even within the same country and the same organization."

Time of day matters too, and differently depending on job length. Dodge and
colleagues evaluated a scheduler allowed to delay a start by up to 24 hours to
catch a cleaner hour. For the short DenseNet runs on digits this cut emissions by
about **26%** on average across regions and months. For jobs longer than a day it
achieved almost nothing, for a reason that is arithmetic rather than engineering:
a multi-day run already averages over the daily cycle, so no peak is left to
avoid. For those, pausing and resuming when the grid is dirty — at the cost of
wall-clock time — is what pays. Note their own caveat: the optimisations were
computed on historical data, so they bound what a real scheduler working from a
forecast could achieve.

The upshot is that **the two largest levers on the carbon cost of a run are where
you run it and when**, and neither is a machine learning decision. Both are
available to a student with a cloud account and a region dropdown, and both dwarf
what any amount of tuning will buy.

---

## 5. Training is usually the smaller half

Almost every circulated figure concerns *training*, because training is a
discrete event with a start and an end, which makes it easy to measure and easy
to dramatise. A deployed model's total energy is training plus every query it
will ever answer:

$$\text{footprint} = \big(E_{\text{train}} + N_{\text{queries}} \cdot E_{\text{inference}}\big) \times \text{CO2e per kWh}$$

and for anything in production the second term dominates. NVIDIA has estimated
that 80–90% of the machine learning workload is inference; Amazon Web Services
has claimed 90% of cloud machine learning demand is. Google published its own
accounting: machine learning was 10–15% of the company's total energy in each of
three consecutive years, split about **three-fifths inference and two-fifths
training**. The estimates differ because the populations do — a cloud provider
sees other companies' serving traffic — but every one puts inference ahead.

This inverts several intuitions. An architecture costing twice as much to train
but 20% less to serve is, on a 10/90 split, a net reduction. A neural
architecture search (NAS) is expensive once and then amortised over every model
trained in that domain afterwards; Patterson's analogy is that the search is
simulating a better light bulb, training is manufacturing the bulb, and inference
is every customer switching it on — so charging the simulation to each bulb is a
category error.

The table in Section 5.5 of the handout therefore has an economic reading nobody
stated in class. "Fit the linear model first" is a claim about accuracy in the
handout — on the digits a hidden layer of 32 was worth 2.8 points, for almost
four times as many parameters, 2,410 against 650 — and a claim about lifetime
serving cost everywhere else. Likewise the table's last row: widening the
hidden layer from 32 units to 64 roughly doubled the parameters, 2,410 to
4,810, and moved validation accuracy from 0.9602 to 0.9611, less than the
spread between runs. Whatever that last fraction of a point is worth, it is
paid for, forever, at every single prediction.

---

## 6. Be careful with other people's numbers

In 2019 Strubell, Ganesh and McCallum published an influential estimate of the
carbon cost of a neural architecture search, arriving at **284 tonnes of CO2e**
for the search that produced one particular model. It was widely reported,
frequently as though it were the cost of training an ordinary model once.

In 2021 Patterson and colleagues — several of them at the organisation that had
actually run the search — recomputed it from internal measurements. The earlier
estimate, they concluded, was **18.7 times too high for an average organisation
and 88 times off in emissions for an energy-efficient one**. Two errors
compounded. The search evaluated candidates on small *proxy tasks* and scaled up
only the winners, a detail not obvious from the original paper, so the estimate
assumed full-size training throughout. And a search is run roughly once per
problem domain, not once per model, so attributing its cost to every subsequent
training run — as much of the coverage did — multiplies the error again.

Read this correctly, because the wrong reading is tempting in both directions.
It is **not** evidence that the concern was manufactured: Strubell and colleagues
raised a question the field was not asking, and the measurement culture that
followed exists largely because they did. Both papers' recommendations converge
on the same fix — stop estimating other people's runs retroactively, measure your
own. Nor is it a story about carelessness; Patterson's paper says the estimate
was made "despite considerable effort."

It is a story about **stacked assumptions**. A retrospective carbon estimate
multiplies a guess at the hardware, a guess at utilisation, a guess at the number
of runs, a PUE and a grid factor. Each is individually defensible and uncertain
by a factor of 1.5 or 2, and five in a row is one to two orders of magnitude of
uncertainty before anyone has made a mistake. Patterson's paper puts the total
spread across model, hardware, facility and location at **100 to 1000 times**,
then draws the corollary: "These large factors also make retroactive estimates of
energy cost difficult."

The advanced version of this lesson, in `Advanced/09_neural_networks_advanced/`,
makes this point with gradient checking, which exists because a wrong gradient
does not raise an exception — it trains, badly, and looks like a modelling
problem. The lighter lesson keeps the habit in miniature: Section 5.2 of the
handout counts the weights by hand, $64 \times 32 + 32$ for the hidden layer and
$32 \times 10 + 10$ for the output, and checks the total against the 2,410 Keras
prints. A wrong carbon estimate does not raise an exception either. It
reads as careful work, propagates into slides and press coverage, and is
corrected years later if at all. The defence is identical: recompute from the raw
inputs by a route that does not reuse the intermediate values.

---

## 7. What is getting better, and what is not

Three trends push the cost down, which is why Patterson's group predicted
training emissions would plateau and then shrink. **Specialised hardware** runs
this workload 2 to 5 times more efficiently than off-the-shelf systems, by
exactly the mechanism Section 2 above gives. **Better models for the same accuracy** —
architecture search, distillation, sparsity, reduced precision — buy equal
quality for less arithmetic. **Cleaner, better-run facilities** are roughly 1.4
to 2 times more efficient on PUE alone, with siting multiplying through the factor of
five to ten in Section 4 above.

Two trends push back, and both are structural. **Diminishing returns:** Thompson
and colleagues, examining a decade of image classification among other
benchmarks, argue each increment of accuracy has been bought with an
exponentially larger increment of compute. Lesson 9 shows the same shape at
laboratory scale — in the acceptance-test sweep of the handout's Section 3.4 the
mean climbed from 0.6048 at one hidden unit to 0.7392 at two, but only from
0.9445 to 0.9549 when eight units became sixteen. The last fraction of a point
is always the expensive one, and industrially "expensive" means a grid connection.
**Embodied carbon:** everything above concerns *operational* energy, but Gupta
and colleagues argue that for modern mobile and datacentre equipment the larger
share of lifetime emissions comes from manufacturing the hardware and building
the infrastructure — a share that grows as operational energy gets cleaner. An
accelerator that halves training energy but is replaced every three years may not
be the improvement the operational figure suggests, and essentially no published
training-cost figure includes this.

---

## 8. What you will be required to report

Reporting has moved, in about six years, from an activist proposal to a legal
obligation.

**In papers.** Schwartz, Dodge, Smith and Etzioni's "Green AI" (2020) proposed
that efficiency be reported alongside accuracy as a first-class result, with
floating-point operations as the hardware-independent measure. Henderson and
colleagues built a framework for systematic energy and carbon reporting; Lacoste
and colleagues built a calculator turning hardware, hours and region into an
estimate; open tooling now logs energy from inside a training script. Major
conferences ask about compute in their submission checklists, MLPerf added power
measurement, and model cards — the documentation format proposed by Mitchell and
colleagues in 2019 — increasingly carry a compute and emissions section.

**In law, for buildings.** The EU's Energy Efficiency Directive (Directive (EU)
2023/1791), Article 12, with Commission Delegated Regulation (EU) 2024/1364,
obliges datacentres with installed information technology power demand of **500
kilowatts or more** to report annually to a European database: total energy
consumption, PUE, water usage effectiveness, energy reuse factor and renewable
energy factor, across roughly two dozen data points, by 15 May for the previous
calendar year. The mechanism is disclosure, not a cap, and its immediate effect
is that PUE and its companions stop being voluntary marketing figures and become
filed numbers.

**In law, for models.** The EU Artificial Intelligence Act (Regulation (EU)
2024/1689), Annex XI, requires providers of general-purpose artificial
intelligence (AI) models to document the **known or estimated energy
consumption** of the model, and says that where it is unknown the estimate may be
based on the computational resources used. The limits matter as much as the
requirement: it covers energy rather than emissions, excludes embodied carbon,
prescribes no methodology, and directs disclosure to regulators rather than the
public.

Within the working life of anyone reading this, "how much energy did that cost?"
moves from a question nobody asks to a field on a form — and the person who fills
it in is the one who ran the job.

---

## What to take from it

- **Scaling is the cheapest lever in Lesson 9 and the most expensive one in
  industry.** Widening the hidden layer from one unit to eight took the
  acceptance test from 0.6048 to 0.9445 and cost seconds on a laptop; the same
  trade at ImageNet scale runs to hundreds of kilowatt-hours, and the published
  tables are the surviving fraction of what was actually run.
- **Moving a number costs far more than computing with it** — about a thousand
  times more for a DRAM access than a floating-point operation. That one ratio
  explains specialised accelerators, reduced precision, the obsession with data
  locality, and why fetching the weights of the handout's 301,066-parameter
  network costs more energy than the arithmetic done with them.
- **The two biggest levers on carbon are where and when you run, and neither is a
  modelling decision.** Grid intensity varies more than fiftyfold between
  national grids and roughly fourfold across one provider's regions, and delaying
  a short job by a few hours cut measured emissions by about a quarter. Serving,
  meanwhile, dominates a deployed model's lifetime — so a slightly worse, much
  cheaper model is often the better engineering decision.
- **Treat any retrospective carbon figure as the product of five uncertain
  assumptions.** The most famous one was later found 18.7 to 88 times too high,
  by people with the actual measurements. Ask which hardware, which utilisation,
  which PUE, which grid, how many runs — and if the paper does not say, the
  number is an illustration, not a measurement.

## Where to look next

| Resource | Type | Why read it |
|---|---|---|
| Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014 | Conference plenary | The price list of Section 2 above, from the person who measured it; it reframes what a fast program is |
| Masanet et al., "Recalibrating global data center energy-use estimates", *Science* 367 (2020) | Paper | 550% more compute for flat energy, 2010–2018 — the best argument against naive extrapolation |
| Patterson et al., "Carbon Emissions and Large Neural Network Training", arXiv:2104.10350 (2021) | Paper | The PUE and grid factors, the training/inference split, and the corrected NAS estimate with its post-mortem |
| Dodge et al., "Measuring the Carbon Intensity of AI in Cloud Instances", FAccT 2022 | Paper | Real measurements across 16 regions and a year, plus the scheduling results of Section 4 above |
| Schwartz, Dodge, Smith & Etzioni, "Green AI", *Communications of the ACM* 63(12) (2020) | Position paper | The case for reporting efficiency as a result, and why floating-point operations were proposed as the measure |
| Gupta et al., "Chasing Carbon: The Elusive Environmental Footprint of Computing", HPCA 2021 | Paper | The embodied-carbon half that operational figures leave out |
| Directive (EU) 2023/1791 Art. 12 and Delegated Regulation (EU) 2024/1364 | Primary legal text | The datacentre reporting obligation in full, at `eur-lex.europa.eu` |
| ISO/IEC 30134-2:2016 | Standard | PUE as actually defined, with the measurement categories that make two quoted figures incomparable |
| IEA, *Energy and AI* (2025) | Official report | The sector-level projections, with their assumptions stated |
| Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, ch. 7, "Domain-Specific Architectures" | Book chapter | Why a TPU exists, derived from the energy numbers rather than asserted |
