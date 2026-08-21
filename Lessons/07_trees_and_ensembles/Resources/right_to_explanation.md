# The Right to an Explanation

> **Supplementary reading — Lesson 7**
> Estimated reading time: 18 minutes
> Not examinable. It is what happens after section 11's leaderboard is handed
> to a lawyer instead of a data scientist.

---

## Why a computer scientist should read this

Section 11 of the handout put every model from this lesson on one table and
drew two conclusions: the ensembles beat any single tree by a point or a
point and a half, and the single depth-tuned tree was "the only model on
this table a person could read start to finish." Read as a modelling
result, 0.883 versus 0.911 is a footnote. Read as a legal and
organisational fact, it is the whole story: a bank that deploys the random
forest instead of the tree has traded three points of accuracy for a
system that, when a customer asks why their loan was refused, has no
sentence to give them — the point section 4 made in passing, "for a
decision that has to be justified to the person it affects, that can
matter as much as accuracy," and pointed here.

Whether that trade is even legally available is a live question in
Europe, argued over by lawyers and computer scientists for the best part
of a decade without a clean resolution — that argument, not a settled
answer, is what this reading covers. Shipping a model that decides
something about a person needs more than "the ensemble was three points
better": it needs knowing roughly where the law's boundary sits, and that
boundary is narrower than most people assume, and disputed at the edges.

---

## 1. Article 22, in outline

The **General Data Protection Regulation** (**GDPR**), the European Union's
data protection law, took effect in 2018. Most of it concerns consent,
storage and access to personal data. One article, 22, targets automated
decision-making specifically, and it is the article this reading orbits.

Article 22(1) gives a data subject — the person the data is about — "the
right not to be subject to a decision based solely on automated
processing, including profiling, which produces legal effects concerning
him or her or similarly significantly affects him or her." A loan refusal
is the textbook example the drafters had in mind; so is an automated
hiring rejection. Article 22(2) carves out exceptions — the decision is
allowed if it is necessary for a contract, authorised by law, or based on
explicit consent — and Article 22(3) says that where an exception applies,
the controller (whoever built and runs the system) must implement
"suitable measures to safeguard the data subject's rights," including "at
least the right to obtain human intervention, to express his or her point
of view, and to contest the decision."

Notice what Article 22 does *not* say. It never uses the word
"explanation." What it grants is a right to contest and a right to a human
in the loop, for a narrow class of decisions that are both fully automated
("solely") and legally or similarly significant. A separate part of the
regulation — Articles 13(2)(f) and 14(2)(g), the notice provisions for
when data is collected — requires the controller to give the data subject
"meaningful information about the logic involved," plus the significance
and envisaged consequences of the processing. This phrase, sitting in a
notice requirement rather than in Article 22 itself, is what the popular
shorthand "right to explanation" is actually built on — and how much
weight it can bear is what the next two sections argue about.

---

## 2. Wachter, Mittelstadt and Floridi: weaker than advertised

In 2017, Sandra Wachter, Brent Mittelstadt and Luciano Floridi published
"Why a Right to Explanation of Automated Decision-Making Does Not Exist in
the General Data Protection Regulation," which became the reference point
for a skeptical reading of what Article 22 grants.

Their argument is a close statutory reading. A binding right to
explanation would need to appear in the operative articles, or at minimum
in a clearly binding recital, and on their reading it does neither:
"meaningful information about the logic involved" is attached to Articles
13 and 14, which govern what a controller discloses *before or at the
point of collection* — an ex-ante notice, not a post-decision account of
why *this* applicant was refused. The one recital that comes closest to
naming an explanation right, Recital 71, is non-binding — recitals explain
the reasoning behind the operative articles but create no obligation of
their own — and its language, a right to "obtain an explanation of the
decision reached," never made it into Article 22's operative text at all,
despite appearing in an earlier draft. Something was written and then
dropped, and the authors read that as deliberate.

Their conclusion is not that Article 22 is toothless: it grants real
safeguards — human intervention, the right to contest, the right to
express a view — but these are *ex-ante* and *systemic*, giving a person
leverage around a decision rather than a guaranteed *ex-post individual*
explanation of their specific refusal. A bank could, on this reading,
satisfy the law by disclosing in general terms that it uses a random
forest trained on income and debt-to-income ratio, without ever being
compelled to say why applicant 812 in particular was refused.

---

## 3. Selbst and Powles: reading it back up

Andrew Selbst and Julia Powles replied the same year, in "Meaningful
Information and the Right to Explanation" — a disagreement about method as
much as about conclusion.

Their objection is that the close-reading approach imports a common-law
style of interpretation — treat only operative articles as binding, read
recitals as advisory colour — onto a civil-law instrument that does not
work that way: courts including the Court of Justice of the European
Union routinely use recitals to resolve an ambiguous operative article.
Recital 71's explicit mention of a right "to obtain an explanation of the
decision reached" is, on their reading, exactly the purposive evidence a
court should draw on when construing "meaningful information about the
logic involved" — read functionally, as whatever information a data
subject actually needs to exercise the right to contest that Article
22(3) undeniably grants. A right to contest with no explanation of what
is being contested is close to hollow in practice, so the two rights
should be read as bound together.

Neither side denies the text is genuinely ambiguous, and no Court of
Justice ruling has settled which reading controls. What the exchange
leaves a practitioner with is a bracket, not an answer: somewhere between
"disclose the method in general terms" and "explain this specific
refusal to this specific applicant," and which end a regulator or court
will enforce remains open years later. Building a system under GDPR with
no explanation strategy at all is a bet on the weaker reading holding.

---

## 4. Where this lesson's models fall on that bracket

This is where sections 4 and 11 of the handout stop being background and
become the concrete case. A shallow decision tree, printed as
`max_leaf_nodes = 9`, is nine leaves and eight thresholds — the handout
called it "an entire model... a sequence of if/else statements a loan
officer could apply by hand." Handed that tree, a bank can answer "why was
I refused?" with an actual sentence: *your income was below 28, and your
debt ratio was above 0.82.* That sentence satisfies even the strong
reading of "meaningful information about the logic involved": specific,
about this applicant, and something a human reviewing it under Article
22(3) can meaningfully agree or disagree with.

A 100-tree random forest, the model section 11 crowned the winner at 0.911
against the tree's 0.883, has no such sentence. Its prediction is a vote
among a hundred trees, each grown on a different bootstrap resample with a
different random feature subset at each split — section 6's whole
mechanism for decorrelating the trees is also what makes any one
applicant's outcome the sum of a hundred separate, individually
defensible but jointly illegible chains of splits. Gradient boosting,
section 11's other strong performer, is worse still: its 30 trees were fit
sequentially, each correcting the pseudo-residual $y - p$ left by every
tree before it, so no single tree corresponds to any interpretable piece
of the final decision. Both ensembles are auditable in aggregate — inspect
the code, check the training data, verify the cross-validated accuracy —
but neither is auditable *per applicant* the way the nine-leaf tree is.

Section 11's recommendation table reached this conclusion from the
modelling side alone, no law invoked: "the decision must be explained to
the person it affects → a single tree, depth-tuned, or nothing on this
list." What this reading adds is that under one plausible reading of
Article 22, that recommendation is not merely good practice for a system
operating in the European Union — it may be closer to a requirement, for
the class of decisions the article actually covers. And the trade on
offer is small in accuracy terms: three points, against a possible legal
argument for or against deploying the model at all.

---

## 5. Explaining a model after the fact anyway

In practice almost nobody chooses the nine-leaf tree — the three-point gap
is real money at loan-portfolio scale. The industry's answer is a second
layer of tooling that produces an explanation for a decision the
underlying model cannot itself state, and it is worth naming what
"explainability" means in a job posting: three specific tools.

**Permutation importance**, which section 7 of the handout already
introduced and already showed can fail, is the crudest of the three:
shuffle one feature's column, measure how much accuracy drops, call that
the feature's importance. Section 7's own experiment is the warning label
— with twenty pure-noise columns added to the loan data, 54% of a random
forest's built-in importance landed on features "wired to nothing,"
because the restricted feature menu at each split sometimes hands a split
to noise for want of anything better. Computed on held-out data rather
than read off training-time impurity reductions, permutation importance
resists that failure better, but it still only answers "how much does
this feature matter, on average, across the whole dataset" — a global
ranking, not a per-decision explanation, and Article 22's safeguards are
about individual decisions.

**LIME** (Local Interpretable Model-agnostic Explanations), introduced by
Ribeiro, Singh and Guestrin in 2016, targets that gap directly. For one
prediction, LIME perturbs the input slightly — nudging income and debt
ratio in many directions — watches how the black-box model's output
moves, and fits a simple interpretable model, typically sparse linear, to
that local neighbourhood: "near this applicant, the forest behaves
roughly like a linear rule that weighs debt ratio twice as heavily as
income." A genuine per-decision explanation, but an approximate and
unstable one — a different perturbation sample can produce a visibly
different local model for the same applicant.

**SHAP** (SHapley Additive exPlanations), introduced by Lundberg and Lee
in 2017, answers the same question with a firmer foundation, borrowing
the Shapley value from cooperative game theory: treat the features as
players cooperating to produce the prediction, and split the credit among
them the one way that satisfies a small set of fairness axioms — each
feature's contribution is its average marginal effect across every
possible order in which features could be "added" to the model. The
output is a signed number per feature per prediction — for one applicant,
perhaps "income: −0.31, debt ratio: +0.44" — summing exactly to the gap
between this prediction and the model's average output. SHAP is close to
a default in industry because it is model-agnostic, carries an axiomatic
guarantee rather than a heuristic, and runs directly against forests and
boosted ensembles of the kind section 11 preferred.

None of the three changes what the model actually computed. They explain
a decision already made by an opaque process, built to be handed to a
customer or regulator after the fact — precisely the arrangement
Wachter, Mittelstadt and Floridi's weaker reading of Article 22 would
permit, and precisely what Selbst and Powles' stronger reading would
demand more of, since a post-hoc local approximation is not the model's
actual reasoning. Whether a SHAP value attached to a random forest's
refusal satisfies "meaningful information about the logic involved" is,
again, not settled.

---

## 6. Credit scoring is not a hypothetical

This lesson's dataset — income, debt-to-income ratio, a default label — is
a small synthetic stand-in for one of the oldest deployed uses of
statistical classification: deciding who gets a loan. Credit scoring
predates machine learning by decades (FICO scores in the United States
date to the late 1950s), precisely because it is an application where a
wrong decision has an obvious victim, a regulator has always been
watching, and the incentive to automate has always been enormous — a human
underwriter reviewing every application does not scale, and a model that
edges out accuracy by a few points is worth real money across a large
portfolio. That combination is why it is the case these authors and the
regulators after them reach for first, and why it is this lesson's
dataset rather than an invented one.

The same shape recurs in hiring — automated résumé screening and
candidate-ranking tools face the same accuracy-explainability tension,
and several jurisdictions (New York City's Local Law 144 among them) now
impose disclosure and audit requirements on them specifically,
independent of GDPR. An opaque ensemble winning by a small margin, and a
post-hoc explanation tool bolted on afterward to satisfy a legal or
reputational requirement, recurs anywhere an automated system decides
something about a person that they can contest.

---

## What to take from it

- **Article 22 grants a right to contest and to human intervention for
  solely automated, significant decisions — its operative text never
  says "explanation."** Whatever explanation right exists is built on
  "meaningful information about the logic involved," a notice-provision
  phrase, plus a non-binding recital.
- **Whether that adds up to a real, per-decision right to explanation is
  a genuine, unresolved dispute**, not one side simply wrong: Wachter,
  Mittelstadt and Floridi read the text narrowly; Selbst and Powles argue
  the interpretive method itself should be broader. No court ruling has
  closed the gap.
- **The accuracy-interpretability trade this lesson quantified — a few
  points, section 11's leaderboard — is not only a modelling choice.** For
  a system deciding something legally significant about a person in the
  European Union it may be a partly legal one, and the accuracy worth
  paying away to stay on the safe side is small.
- **SHAP and LIME are the two tools an employer will expect you to know
  by name**; permutation importance, already shown in section 7 to be
  foolable, is the crudest of the family and no substitute for either.

## Where to look next

- **Regulation (EU) 2016/679**, Article 22 and Recital 71, and Articles
  13(2)(f) / 14(2)(g) — the text everything above argues about, in full
  at `eur-lex.europa.eu`.
- **Wachter, Mittelstadt and Floridi (2017)**, "Why a Right to Explanation
  of Automated Decision-Making Does Not Exist in the GDPR," *International
  Data Privacy Law*, 7(2).
- **Selbst and Powles (2017)**, "Meaningful Information and the Right to
  Explanation," *International Data Privacy Law*, 7(4) — the direct reply.
- **Lundberg and Lee (2017)**, "A Unified Approach to Interpreting Model
  Predictions," NeurIPS — the original SHAP paper.
- **Ribeiro, Singh and Guestrin (2016)**, "'Why Should I Trust You?':
  Explaining the Predictions of Any Classifier," KDD — the original LIME
  paper.
