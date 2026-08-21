# Segmentation and the Ethics of Inferred Categories

> **Supplementary reading — Lesson 8**
> Estimated reading time: 25 minutes
> Not examinable. The distinction in Section 2 — that an inferred category has
> no ground truth to be wrong against — is not mathematics, but it is the kind
> of thing you should not meet for the first time in a product meeting.

---

## Why a computer scientist should read this

Section 3.3 of the handout ran DBSCAN on Aurora's session data and, as a
side effect of finding the bot cluster, produced thirteen sessions the
algorithm would not put in either group: **noise**, in the technical
sense of "not confidently either." The handout checked their true label —
because this is a lesson and the data is synthetic — and reported that
all thirteen were genuine humans, customers whose one ordinary week
happened to look, by chance, almost as mechanically regular as a
script's. That check is a luxury a real bot-detection pipeline never
has. Somewhere behind Aurora's thirteen noise points sits a decision — a
CAPTCHA (Completely Automated Public Turing test to tell Computers and
Humans Apart) challenge, an order hold, a rate limit, or nothing — made
about a real person on the strength of a label nobody can confirm.

Section 2 of the handout ran k-means on the same retailer's customers and
recovered four segments — "window shoppers," "bargain hunters," "occasional
big spenders," "loyal high spenders" — that the handout could grade against
`retail_data.py`'s true generating groups (an adjusted Rand index (ARI)
of 0.985) precisely because this lesson chose to publish a ground truth
that a real marketing team never has. Take that asymmetry out and put a person on the
receiving end of the four labels — a loyalty tier, an offer, a credit
limit, a price — and the two notebooks stop being a clustering exercise
and become a case study in a much older problem: what an organisation is
allowed to infer about you, from what, and what happens when the
inference is wrong and nobody can say so. This reading is not about
clustering being unusually dangerous; it is about what changes, legally
and practically, the moment the *purpose* of a method built to find
whatever structure is there — with no answer key to consult, ever —
shifts from "understand our customers" to "decide something about this
one."

---

## 1. Segmentation is one of the oldest deployed uses of this family of methods

Before k-means had a name, retailers were segmenting customers by hand:
recency-frequency-monetary (RFM) scoring, drawn straight from a customer's
purchase history, has been standard direct-marketing practice since the
1990s and predates any of this lesson's algorithms in industrial use.
Loyalty programmes — a supermarket card, an airline mileage tier, a
coffee-shop app — exist substantially to generate the transaction history
segmentation runs on; the point-of-sale discount is, from the retailer's
side, the price of the data. What k-means and its relatives added was
scale and automation: instead of an analyst hand-building three or four
customer archetypes from intuition, a clustering algorithm finds however
many groups the data actually supports, continuously, across millions of
customers, with no person looking at any individual row.

The most-told version of what this makes possible is the 2012 case of
the American retailer Target, reported at length by Charles Duhigg in
the *New York Times*: a statistician built a model — the same family of purchase-pattern
analysis this lesson's section 2 uses — that inferred pregnancy from
shifts in buying behaviour well before a customer had told anyone, and
Target began mailing baby-product coupons accordingly. What made the
story travel was not the model's accuracy but a father who complained
that his teenage daughter was being sent baby coupons, then called back
days later to apologise — she was in fact pregnant and he had not known.
Nothing about the story requires the model to have been wrong. It
requires only that an inference, made from data given up for an
unrelated reason, reached a parent before its subject chose to share it.

The same shape now runs two markets beyond retail marketing. **Credit and
insurance underwriting** cluster applicants or policyholders into risk
tiers from behavioural signal — a telematics device scoring driving style
into a usage-based insurance premium — and a tier, unlike a marketing
segment, routes directly to a price. **Ad targeting** builds "lookalike"
audiences by clustering users who behave like an advertiser's existing
customers, at a scale — hundreds of millions of people, refreshed
continuously — that makes any individual customer's presence in a given
audience essentially invisible to them. In every one of these, the
mechanism is the same one section 2 walked through by hand: assign every
point to its nearest representative, update the representative, repeat.
What differs is only what happens after the label is assigned, and
Sections 3 through 5 are about exactly that difference.

---

## 2. An inferred category has no ground truth to be wrong against

Section 4 of the handout drew a careful line between **internal** metrics —
the silhouette score, computable from the clustering and the data alone —
and **external** metrics — the adjusted Rand index, computable only because
this lesson happens to know `retail_data.py`'s `true_segment` column.
Aurora's actual analysts, the handout said plainly, would have the
clustering and nothing to check it against.

Sit with what that means once the clustering leaves the notebook. A
supervised classifier trained to predict "will this loan default" is wrong
in a sense a court, a regulator, or the model's own author can state
precisely: the loan defaulted and the model said it would not, or vice
versa, and the ground truth arrives eventually. "Window shopper," "bargain
hunter," "loyal high spender" are not that kind of claim. They were never
predictions of an external fact that will later be confirmed or refuted;
they are *names Aurora chose* for regions of a scatter plot that k-means
carved up on the strength of two numbers. There is no future event that
reveals a customer's "true" segment, because no such thing exists
independently of the clustering that produced it. This is not a
shortcoming of k-means or of this dataset — it is definitional.
Unsupervised methods do not predict a label that already has a meaning;
they *manufacture* the label, and the label's only claim to correctness
is that it is a self-consistent, reasonably compact grouping of the data
it was run on — precisely what an internal metric like the silhouette
score checks, and all it checks.

The practical consequence is that "the model is wrong" stops being a
question with a computable answer. For an inferred segment, the only
things that can be said are weaker and more contestable: the grouping is
*unstable* (a different sample of customers, or a different value of $k$,
carves the boundary somewhere else); the grouping is *not useful* for the
purpose it is put to (a "low-value" segment mostly containing customers
who would spend more given a better offer, which the clustering has no
way to know, because it was never told what would have happened under a
different offer); or the grouping *causes harm when acted on*, a claim
about consequences, not correctness. All three are judgements a person
has to make, informed by the data but not settled by it — exactly what
section 4's closing point about internal metrics warned against treating
as a general endorsement: a clustering can be self-consistent and still
be the wrong thing to build a decision on top of.

---

## 3. Proxy discrimination: recovering a protected characteristic nobody asked for

Aurora's clustering pipeline, as built in this lesson, never sees a
customer's race, sex, disability status, religion, or any of the other
categories that anti-discrimination law in most jurisdictions singles out
for protection. Section 2's k-means sees two numbers — annual spend and
visit frequency — and section 3's DBSCAN sees two more — session duration
and pages viewed. Neither algorithm has any mechanism for using a
protected characteristic, because neither is given one.

That is not the same as the resulting clusters being independent of
those characteristics, and the gap between the two is what **proxy
discrimination** names. Spend and visit frequency, or their real-world
analogues — postcode, device type, time of day, which product categories
a basket contains — routinely correlate with protected characteristics
for reasons that have nothing to do with the algorithm: residential
segregation makes postcode a strong proxy for race in many countries;
short, interrupted browsing sessions can correlate with disability or
caregiving; certain product categories correlate with gender or
pregnancy, as the Target story shows directly. A clustering algorithm
handed only "innocuous" features can reconstruct, as an emergent
property of the geometry it finds, a grouping that lines up closely with
a protected characteristic — not because it was told to, but because
that characteristic was one of the real causes of the pattern in the
innocuous features. Section 5's own eight-column account table makes the
mechanism vivid even though its columns are not sensitive: three latent
factors — spending propensity, engagement, price sensitivity — generate
all eight observed columns, so knowing the eight lets principal
component analysis (PCA) recover the three almost exactly. Nothing stops a latent factor behind a real
customer table from being, in part, a protected characteristic never
collected as such.

This is a genuine bind, not just a risk to be careful about. Checking
whether a clustering has proxy-discriminated — the unsupervised analogue
of lesson 4's confusion-matrix-per-group check — requires knowing group
membership on the very characteristic the clustering was built without.
Under European Union (EU) law (Section 5), that characteristic is
usually **special category data** — the term used by the General Data
Protection Regulation (GDPR), the EU's data protection law, for data
revealing racial or ethnic
origin, political opinion, religion, trade union membership, genetic or
biometric data, health, or sexual orientation — restricted by default,
with only narrow exceptions. Auditing a segmentation for exactly this
failure may need collecting the one category of data an organisation is
otherwise discouraged, or in some cases not permitted, to hold. There is
no tidy resolution offered here; Barocas and Selbst's paper in the
further reading works the tension through properly.

---

## 4. Two individual harms this lesson's own notebooks produced

Two concrete failure modes are already sitting in the handout, not invented
for this reading.

**Being the wrong kind of ambiguous.** Section 3.3's thirteen noise
points are DBSCAN's honest admission that it cannot confidently place
those sessions in either the bot or the human cluster — the handout
calls this "a more honest output than a forced binary label would be,"
and it is, *as an output*. What it becomes downstream is a separate
question the handout never has to answer, because the lesson stops at
the notebook. A rate limiter, a step-up authentication challenge, an
account hold, a manual review — any of the ordinary things a
bot-detection pipeline does with an ambiguous verdict costs the thirteen
genuine customers behind it time, friction, or a blocked purchase, for
behaving, by chance, a little more consistently than usual for one week.
DBSCAN did not misclassify them — noise is a legitimate, well-defined
output, distinct from a wrong label. But a system built on top of it has
to decide what "not confidently either" *means* for the person on the
other end, and "possibly a bot" is a very different consequence from
"possibly a very consistent human," even though the algorithm's evidence
for both is the identical sparse neighbourhood.

**Being the low-value segment.** Section 2's four true customer groups
include "window shoppers" — 500 of Aurora's 2,000 customers, built with
a mean annual spend of €80 and 1.4 visits a month, the smallest spend
and lowest engagement of the four segments the notebook recovers.
Nothing in the notebook says what Aurora does with that label, because
the lesson stops at discovering it — but the obvious commercial use of a
low-value segment is to spend less marketing budget on it, offer worse
loyalty terms, or deprioritise its service queue, on the entirely
reasonable logic that resources are finite. The individual harm is not
that the segmentation is inaccurate — by construction here it is nearly
perfect, ARI 0.985 — but that being correctly placed in "window
shoppers" can become a self-fulfilling floor: worse offers produce less
reason to spend more, which confirms the segment, which justifies the
next round of worse offers. A supervised model predicting churn at least
has a future outcome that can eventually contradict it. A marketing
segment, absent a deliberate experiment offering the "wrong" treatment
to a control group, has no such correction built in — a direct
consequence of Section 2's point: there is no outcome waiting to prove
the inferred category wrong.

---

## 5. What the regulation actually says

Three provisions of the GDPR bear directly on Aurora's two notebooks —
lesson 7's resource,
`07_trees_and_ensembles/Resources/right_to_explanation.md`, covers
Article 22 in full and is the place to go for depth beyond this summary.

GDPR Article 4(4) defines **profiling** as any automated processing of
personal data used to evaluate personal aspects of a natural person — in
particular to analyse or predict economic situation, preferences,
interests, behaviour, or location. Section 2's customer segmentation and
section 3's bot detection are both, in the regulation's own vocabulary,
profiling, whether or not anyone building them thinks of the word as
applying to marketing analytics.

Article 21(2) gives a data subject the right to **object, at any time and
without justification**, to processing of their data for direct
marketing purposes — a category that explicitly includes profiling to
the extent it relates to that marketing. Unlike Article 22, which
protects only decisions "solely" automated with a legal or similarly
significant effect, Article 21(2) has no such conditions and no
exceptions available to the controller: once a customer objects, their
data must stop being used for marketing, full stop. Section 2's
clustering — used, in the handout's own framing, "for a marketing
campaign" — sits squarely inside this provision. Article 22 itself
addresses a narrower, separately governed situation: a decision based
*solely* on automated processing that produces legal effects or
similarly significantly affects the person, a rejected credit
application being the standard example. Whether an automated fraud hold
following a DBSCAN noise verdict counts depends on how automated the
downstream decision is and how significant its effect is judged to be —
questions lesson 7's resource shows are genuinely disputed even for a
squarely covered case like a loan refusal, and no less disputed here.

Recital 71 — non-binding, as lesson 7's resource explains recitals are,
but used by courts to interpret the operative articles — says a
controller should use appropriate mathematical and statistical
procedures for profiling and take measures that prevent discriminatory
effects on the basis of the special categories listed in Section 3. It
is the closest the regulation comes to naming, directly, the
proxy-discrimination problem Section 3 describes — without using that
term, and without resolving the tension Section 3 raised, that checking
for such an effect usually requires the very data the regulation
otherwise restricts. None of this settles what Aurora must do; it fixes
the vocabulary a real pipeline has to be checked against.

---

## What to take from it

- **An inferred category is not a prediction of an external fact — it is
  a name chosen for a pattern the algorithm found**, so "the model is
  wrong" has no computable meaning the way it does for a supervised
  label. What can be assessed is whether the grouping is stable, useful,
  and harmless when acted on — none of which a silhouette score answers.
- **A clustering built entirely on innocuous features can still recover a
  protected characteristic**, because both can share an underlying
  cause. Checking for this usually requires collecting the very category
  of data privacy law restricts by default — a real bind, not a solved
  problem.
- **DBSCAN's noise bucket and k-means' lowest-value segment are both
  legitimate, correct algorithm outputs that still cost a real person
  something once acted on** — friction for the thirteen genuine sessions
  of section 3.3, worse terms with no natural correction for the window
  shoppers of section 2. The harm sits in what was built on the output,
  not in the output being wrong.
- **GDPR gives marketing profiling an unconditional objection right
  (Article 21(2)) and gives solely automated, significant decisions a
  narrower, still-disputed set of protections (Article 22)** — which one
  applies depends on what the segmentation is used to do, not on which
  algorithm produced it.

## Where to look next

| Resource | Type | Why read it |
|---|---|---|
| Regulation (EU) 2016/679, Articles 4(4), 21, 22 and Recital 71 | Primary legal text | The provisions Section 5 summarises, in full, at `eur-lex.europa.eu` |
| Barocas & Selbst, "Big Data's Disparate Impact," *California Law Review* (2016) | Paper | The proxy-discrimination mechanism in Section 3, and the audit-data bind, argued rigorously |
| Duhigg, "How Companies Learn Your Secrets," *New York Times Magazine* (2012) | Long-form journalism | The Target pregnancy-prediction case in Section 1, told in full |
| Lesson 7's supplementary reading, *The Right to an Explanation* | This course | Article 22 and the explanation debate, in the depth Section 5 only summarises |
| Barocas, Hardt & Narayanan, *Fairness and Machine Learning*, chapter on unsupervised and representation-level fairness | Book | The formal treatment of proxy variables and fairness without ground-truth labels |
