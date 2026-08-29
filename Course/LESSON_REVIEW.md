# Reviewing a finished lesson

`tools/verify_lesson.py` decides whether a lesson is *built*. This document is about
whether it is *right*, which is a different question and is not automatable in full.

Every check below earned its place by catching something real in lesson 1 during the
review of 28 August 2026. The examples are kept deliberately: a check without a
failure it once caught tends to be skipped.

Work in this order. The mechanical checks are cheap and narrow the field; the
judgement ones need the built PDF open in front of you.

---

## 1. Mechanical

### 1.1 No figure sits directly under a heading

Every figure must follow the paragraph that introduces it, so a reader meets the
argument before the picture. A section that opens with a stack of images is a section
whose figures were never placed.

> **Caught in lesson 1:** thirteen of twenty-one figures. Section 7 opened with five
> pictures in a row, then the prose explaining them.

```bash
python tools/review_lesson.py NN        # reports every figure under a bare heading
```

### 1.2 Every number in prose appears in a committed notebook output

The handout and slides quote numbers the notebooks produce. Nothing but this check
ties them together — `verify_lesson.py` compares `worked_examples.py` against the
handout, and both can agree while the notebook beside them says something else.

> **Caught in lesson 1's neighbours:** lesson 2's handout said AUC 0.752 where its
> notebook had long printed 0.751, and lesson 3's headline 247,514 had become
> 3,097,038,010 under the shipped image. Both lessons passed verification throughout.

### 1.3 Notation follows the table in CLAUDE.md

`m` is the number of examples, `n` the number of features, `α` the learning rate, `λ`
the penalty. Slips are invisible one document at a time.

> **Caught in lesson 1:** slide notes said the standard error "falls as one over root
> n" where they meant `m`; the quiz wrote the empirical risk as `(1/n) Σ`.

### 1.4 No em dashes in anything projected

Slide bodies, slide titles, metadata — and **inside figures**, where the text is drawn
into the PNG and no search of the markdown will find it. Handouts keep theirs: an em
dash in a typeset PDF is ordinary publishing.

> **Caught across the course:** 839 in the decks, then a further 41 drawn inside
> figures, which the first pass could not reach.

### 1.5 The lesson plan's slide ranges match the segments

`verify_lesson.py` checks the cited slides *exist*, not that they are the right ones.

> **Caught in lesson 1:** the plan cited 9–22, 23–30, 31–39, 40–51 for segments that
> actually began at 21, 31, 37 and 50.

### 1.6 Every notebook is cited in the handout body, with its numbers

Not only in the lesson plan table. A notebook nobody quotes is a notebook nobody runs.

> **Caught in lesson 1:** notebook 02 appeared in the plan table, one homework line and
> three sets of speaker notes — and nowhere else. Handout section 5 had no numbers at
> all, the only section in the lesson arguing entirely in the abstract.

---

## 2. Judgement — the built PDF open, not the markdown

### 2.1 Open every figure and check the caption describes *that* image

The most productive check in the whole list, and the one no tool can do.

> **Caught in lesson 1:** three captions described figures that had never been made.
> `error_costs.png` was promised as "two problems side by side" and shows one.
> `overfitting.png` was captioned as two error curves and shows three polynomial fits.
> A slide's notes said "follow the two curves" of a histogram.

### 2.2 Speaker notes point at the slide they are on

A note describing a figure that is on the *next* slide is the standard failure, because
notes get appended while the author is thinking about what comes after.

> **Caught in lesson 1:** five of them.

```bash
python tools/review_lesson.py NN        # flags deictic notes with no figure beside them
```

### 2.3 Slide titles are true, and carry the argument

> **Caught in lesson 1:** "The same idea, on real data" over a figure fitted to 22
> synthetic points. Retitled to what its own notes tell the lecturer to point at.

### 2.4 Every term is defined before it is used

Students cannot look it up mid-lecture. Expand acronyms, and define words that look
ordinary but are not.

> **Caught in lesson 1:** `R²` used four times and never defined — nor anywhere else in
> the ten lessons. `n_init=10` with no explanation. "cultivar" used seven times.

### 2.5 No claim promises more than the cell delivers

> **Caught in lesson 1:** the self-supervised section said the learned representation
> "transfers to tasks you do care about" and demonstrated no transfer at all. The
> shortcut-feature section said the column "is empty" at deployment, implying an error
> that in practice never comes.

### 2.6 Criteria are stated at the moment they are applied

A test the reader can only run after the fact is not a criterion.

> **Caught in lesson 1:** "errors are catastrophic and unexplainable" as a reason *not
> to build* a model described a model already deployed and already failing.

### 2.7 Look at the rendered deck

```bash
soffice --headless --convert-to pdf deck.pptx
pdftoppm -png -r 55 deck.pdf slide
```

Overlapping annotations, a legend across a curve, a cropped axis label, a title
wrapping to two lines: none of these fail a check, and all are obvious in the render.

---

## 3. The notebooks, which everything else quotes

Sections 1 and 2 check the material *around* the code. This one checks the code, and
it is the pass that matters most, because the handout, the slides and the quiz all
quote numbers these cells produced. A wrong cell is wrong in five places at once, and
`verify_lesson.py` will pass throughout — `worked_examples.py` recomputes what the
*handout* prints, so a handout faithfully reprinting a bad notebook number agrees with
itself perfectly.

Run every notebook in the container, then work the list below.

### 3.1 The cell must compute the thing the sentence describes

Not something close to it. A proxy that lands near the right answer is the hardest
version of this to catch, because nothing looks wrong.

> **Caught in lesson 2:** the lesson's carry-home number, "98 of 128 imputed rows
> borrowed from the test set", came from a neighbour search over 3 standardised
> columns. The `KNNImputer` that actually imputed used 4 raw columns, the
> `nan_euclidean` metric, and only rows that had an `age` to donate. Measuring what
> the sentence claimed gave **94**.

### 3.2 A quoted number must be a property of the run, not of where it stopped

Before reprinting a value, ask what it would be after twice as many iterations.

> **Caught in lesson 2:** "the cost reaches 3.34, far above where it started. It
> diverges." The loss does not diverge — it oscillates between 0.69 and 8.29 and stays
> there, and 3.34 was simply where the bounce happened to be at step 200. The mean over
> the first hundred steps is 3.64 and over the second hundred 3.04; at 2000 steps it is
> still in the same band.

### 3.3 Dead code is evidence about what the author expected

A branch that cannot fire usually marks a belief the data did not support.

> **Caught in lesson 2:** an `if len(losses) < n_iter` branch printing "diverged after
> N steps" — unreachable, because the run never breaks early. It was written by someone
> who expected a divergence that does not happen, and it sat directly above the prose
> claiming one.

### 3.4 Each method's own preconditions apply to it, including the lesson's

The lesson teaches a rule in section 5 and then breaks it in section 9 with no comment.

> **Caught in lesson 2:** `KNNImputer` fitted on unscaled columns, where the standard
> deviations run 184, 54, 13 and 1.1 — so "nearest neighbour" meant "similar bill" and
> `num_support_calls` contributed nothing. Section 5 of the same lesson is the argument
> that distance-based methods need scaling first.

### 3.5 A number that motivates a technique must not be an artefact the lesson removes

> **Caught in lesson 2:** scaling was motivated by `monthly_charges` having "3-4 times"
> the spread of `tenure_months`. That ratio is 3.6 only because of the 20 billing errors
> section 4 exists to find; cleaned, it is **0.8**, the other way round. The slide notes
> said 0.8 and the notebook said 3-4, and neither knew about the other.

### 3.6 Every API new to the lesson is named in prose, and linked

Students cannot look up a call they have only seen inside a code cell, and the
conceptual explanation being good is not a substitute — it tells them what is
happening, not what to type into a search box.

> **Caught in lessons 1 and 2:** zero documentation links across all six notebooks.
> `train_test_split`, `make_pipeline`, `predict_proba`, `cross_val_score`, `f_classif`,
> `get_dummies` and `roc_auc_score` appeared only in code, several of them on first
> encounter in the whole course.

### 3.7 Re-run in the container, and check the figures did not move

Byte-identical figures confirm the committed images came from the same environment.
Anything that moves without an edit to the plotting code is a reproducibility problem
worth understanding before it reaches a student.

### 3.8 Run every code block the prose prints, exactly as printed

A block that carries its own `import` lines is a promise that it is complete. Paste it
into a fresh interpreter and see. The same goes for the loading snippet an exercise
hands out: if the prose describes the data, the snippet has to produce that data.

> **Caught in lesson 2:** the handout's `ColumnTransformer` block used
> `numeric_columns` and `categorical_columns`, defined nowhere, and called
> `LogisticRegression` without importing it — three `NameError`s in a block printing
> its own imports.

> **Caught in lesson 1:** the exercise's `fetch_openml` call returns columns named
> `V1...V11` and quality scores renumbered **1–7**, while the brief promises eleven
> named physico-chemical measurements and "an integer from 3 to 9" and asks students to
> justify a threshold on that scale. Nothing errors; the assignment is simply written
> against data the snippet does not load.

### Log

What has actually been through this pass, so nobody redoes it or assumes it was done.

| Lesson | Date | Result |
|---|---|---|
| 1 | 28 Aug 2026 | **Arithmetic clean.** Every headline number recomputed independently and reproduced exactly: baseline 0.629 and model 0.986 on breast cancer; wine 0.981, $R^2$ 0.622 and 0.816, adjusted Rand index 0.897; the `n_init=1` study (6 distinct solutions, 0.322–0.915); the 30-seed split spread 0.917–1.000 and 5-fold 0.960 ± 0.030. Fixed: no documentation links anywhere and seven APIs never named outside code cells; one imprecise comparison ("a little over one point", "five times") replaced with the measured 1.4–2.3 and 6.8–8.1. Code blocks (3.8): the handout has none; the exercise's loader contradicted its own brief, and the setup slide's `git clone` wrapped twice — the second wrap inside the URL with no backslash, so anyone copying it got a broken command. |
| 2 | 28 Aug 2026 | **Three real errors**, all listed above as 3.1–3.5: the 98-of-128 proxy, the false divergence claim with its dead branch, and `KNNImputer` on unscaled columns. Carry-home number moved to 94 of 128 and propagated to handout, slides, quiz, figure and `CLAUDE.md`. Notebook 1's arithmetic was clean. Code blocks (3.8): three undefined names in the handout's pipeline block. |
| 2 (2nd pass) | 28 Aug 2026 | **Worked the exercise as a student, on seed 104.** Tasks 1–5 and 6(b) are sound. Task 6(a) was a trap: over 15 seeds the imputation leak's effect on AUC has mean +0.0001 and comes out **negative in 7 of 15**, while Task 7 asked students to explain why the numbers differ, with 30% of marks on that explanation. Neither lever the material offered opens the gap — `age` does not appear in the generator's churn model at all, so contaminating it cannot move a score. Rewrote 6(a), Task 7, the hints and the rubric around that finding; corrected the same false promise in handout §9.1 and in a "Try this" this reviewer had introduced earlier the same day. Cross-lesson references: all resolve, one overstatement fixed. Quiz: numbers current, §7 had no question at all — added. Resources: three GDPR list omissions. Speaker-note timings realigned to the plan. |
| 3 | 28 Aug 2026 | **Complete.** Section 3.2 claimed notebook 1 "does the same thing on 450 houses and gets 2,410" — a number in no notebook. The same one-feature fit gives **2,785**, further from the truth than the three-house hand example, because area absorbs `bedrooms` (+0.77) and `bathrooms` (+0.49); all six features give 2,421. Notebook 2 still described a "degree-15 fit on 84 training points" where it fits degree 12 on 21 — one of the four errors CLAUDE.md already records as historic. Section 4.4 said "design matrix" for a table computed without the intercept column, where its own Section 3.1 defines the design matrix as including one (654, not 285). The deferred conditioning paragraph is written, with the ridge conditioning table recomputed from singular values — the eigenvalues of X'X are numerical noise at 4.2e9, which would have been an unfortunate way to get a number wrong in a section about conditioning. Fifteen figures moved under the prose that introduces them. `worked_examples.py` from 17 checks to 37. Quiz: question 17 was built on pre-restack numbers (247,514 / 182 / 16.3, none of which exist anywhere any more) and the quiz used **α for both the learning rate and the penalty, four questions apart** — the exact collision CLAUDE.md's notation table names lesson 3 for. Slides: the same 2,410 claim, and "118 at degree 9, 182 at degree 12" where the table now reads 590 and 24,656. Exercise: the brief called an hourly dataset "daily hire counts" (17,379 rows, an `hour` column), and used α for the penalty throughout; worked through end to end, and the ridge sweep is flat across five orders of magnitude, so the brief now says so rather than letting students tune until the number moves. Resources checked, citations sound, nothing to change. |
| 4 | 28 Aug 2026 | **Complete, and the cleanest so far.** Every worked number in the handout recomputes: the cost arithmetic (140/2740 = 0.051, 13×140 + 33×2600 = 87,620), the odds multipliers as exp of the fitted coefficients, σ(−6.09) = 0.0023, and the AUC identity checked against 200,000 sampled pairs. The 33-question quiz is accurate throughout, including its harmonic-mean arithmetic. Two structural fixes: thirteen figures placed under the prose that introduces them, and `precision_recall_curve.png` moved out of §6.1 — it plots precision against recall with thresholds marked, and sat two sections before any of those three words was defined. Section 6 opened straight onto a subsection and now has an introduction. Exercise worked end to end: the decoy is identifiable (−0.005 against a true 0), theoretical and empirical thresholds agree to 0.039 vs 0.045, and the imbalance test delivers its point cleanly — AUC 0.931 → 0.922 while precision falls 0.256 → 0.078. Resources checked; references sound, nothing changed. |
| 5 | 28 Aug 2026 | **Complete.** No wrong numbers in any artefact — every figure the handout quotes traces to a notebook, and the 29-question quiz is accurate. The gap was verification: `worked_examples.py` had six checks, the fewest in the course, for the lesson that is *about* verification. Now nineteen, with section 2.2's whole table (0.885 / 1.000 / 0.955 / 0.024, ten of two hundred above 0.99) rebuilt from the generator rather than transcribed. Eight figures placed under their introducing prose. The exercise was worked end to end and is the best-designed in the course so far: three planted flaws, of which the two everyone predicts are worth 0.006 between them and the third — 12 readings per turbine split at random — is worth 0.037 and multiplies the fold spread by six. Resources checked; references sound. **One finding for the whole repository:** `StratifiedGroupKFold` assigns groups differently between scikit-learn 1.3.1 and 1.9.0, so this exercise's honest figure is 0.881 in the image and 0.868 outside it. Verification done in the host venv is not verification. |
| 6 | 28 Aug 2026 | **Complete.** The γ/C table in section 5.4 — six numbers the handout and the slides both print — was computed only inside three figure titles and printed nowhere, so nothing compared it against anything. All six are correct (verified against the notebook's own fold seed; a different seed gives 0.930 and 0.893, which is how the gap showed up). Notebook 3 now prints the table, and `worked_examples.py` recomputes it: 20 checks to 26. Section 4.5's "reporting 0.999" read as a measurement of this dataset sitting beside two real ones; it needs correlated features, which the pump data has none of, so it now says so and gives the conditions under which it happens (ten near-copies of one column: accuracy 0.75, two thirds of predictions above 0.999). No figures out of place. |
| 7 | 28 Aug 2026 | **Complete, no wrong numbers.** The depth sweep, bagging, boosting and the 1/e bootstrap limit all recompute against the notebooks. The final leaderboard's second column — each score as a fraction of the 0.93 noise ceiling — is arithmetic the handout does in prose and nothing checked; now gated. 38 checks to 46. |
| 8 | 28 Aug 2026 | **Complete, no wrong numbers.** 1,500 sessions with 180 bots leaves the 1,320 humans the text quotes, DBSCAN's 13 noise points really are all human, and the scree eigenvalues sum to the 8 standardised columns. Section 5.4 opened on the scree plot before saying what a scree plot is for; it now has an introduction and the figure follows the table. |
| 9 | 28 Aug 2026 | **Complete, and careful.** No wrong numbers, and the handout does something none of the others do: it quotes the *median* end-to-end gradient ratio (3,547) across eight seeds, states the 2,734–4,607 range, and warns in the text against quoting the largest as though it were a law. Six figures placed under their prose; section 11.2 opened on two figures back to back and now says what question they answer. One speaker note said "the figure is in handout section 7.4" with a different figure on the next slide — disambiguated. |
| 10 | 28 Aug 2026 | **Complete, no wrong numbers.** Every parameter count checks: 576×256+256 = 147,712 dense against 8×9+8 = 80 convolutional, so the slides' "147,632 missing parameters" and the ratio of 1,846 are both exact. Section 3.3 opened on a figure; it now says why one writes kernels down before learning them. |

### What check 2.1 found in lesson 3

Recorded because the faults have a shape, and it is not the one the check was
written for. Only one was a caption describing the wrong picture; three were
figures that did not show what they claimed, and the worst was a stale number
nobody could see.

- **`alpha_trade_off.png` had its six points hard-coded, and one was pre-restack.**
  The leftmost said test error 182.0 where the truth is 17.3 — and 182.0 is one of
  the two values removed from the quiz the same morning. That single number
  manufactured the entire left arm of the U and put the marked optimum in the wrong
  place: the real minimum is λ = 10⁻⁴, not 0.01. The figure now computes its own
  sweep over ten orders of magnitude.
- **`learning_rate_regimes.png` contradicted the rule its own section derives.**
  Section 4.3 states convergence requires α < 2/c, the bowl has c = 1, and the panel
  titled "Too large: it climbs out" used α = 1.96. It converged. Now 2.15.
- **`train_test_by_degree.png` showed none of what its caption promised.** On a
  linear axis the degree-12 value of 24,656 flattened every other point onto the
  x-axis, so neither the falling training error nor the turn at degree 3 was
  visible. Log scale.
- **`coefficient_trust.png` carried a caption for a different figure** — "the same
  coefficient across four random splits", where the image is six features' error
  against their correlation with `area_sqm`. Two of its labels also overlapped.
- **`regularisation_paths.png` and `alpha_trade_off.png` labelled the penalty α.**
  CLAUDE.md reserves α for the learning rate. The quiz and exercise were corrected
  earlier the same day; the figures draw it into the PNG, where no search finds it.
  `regularisation_paths.png` also titled a panel "everything shrinks" over a curve
  that triples first.

### What check 2.1 found in lessons 4 to 10, and lesson 1

Worth recording because the result is mostly negative, and a negative result from
this check is the expensive kind to obtain.

- **No caption anywhere in lessons 4, 6, 7, 8 or 10 misdescribes its figure.** Every
  number drawn into those images traces to a notebook: lesson 4's three ROC operating
  points and its multiclass matrix whose rows sum to the classification report's
  supports; lesson 6's three support vectors, all three of which sit at |decision| =
  1.000 exactly; lesson 7's box plots and bootstrap limit; lesson 8's elbow and
  silhouette both landing on k = 4; lesson 10's permutation bars.
- **Lesson 5 needed two.** Its high-variance learning curve was captioned "still
  climbing" where it rises to 250 examples and then flattens, and section 6.1 built
  its advice on that slope. Its bias-variance figure was captioned "bias falls with
  complexity" over a curve that climbs four orders of magnitude — which the prose
  three paragraphs below already explained, so the caption was asserting what the
  text had to walk back.
- **Four legend or label placements**, in lessons 3, 4, 9 and 10, each sitting over
  the curve or bar the figure exists to compare.
- **One cross-lesson attribution**, found while checking item 6: lesson 10 called its
  dense baseline "lesson 9's dense network on the same images: two layers of 256".
  Lesson 9's digit networks are one and two hidden layers of 64 units on 8×8 images.
  The 213,761 parameters are right for lesson 10's own 24×24 images; the attribution
  was not.

### What check 2.7 found: the decks, looked at rather than counted

Added 29 August 2026, after rendering all ten decks to one PNG per slide and
reading them. Three faults, none of which any existing check could see.

- **Twenty slides ran off the bottom of the slide**, all in lessons 7 to 10 and
  all the same shape: five bullets that each wrap to two lines. The last line
  either sat behind the university crest or was cut off by the edge. The check
  in `postprocess_pptx.py` was supposed to catch this and did not - it counts
  characters, and scored all thirteen of the worst slides *identically* before
  and after they were fixed. `verify_lesson.py` now rasterises the built PDF and
  looks for ink below the footer line, which is the thing a student sees. On the
  previous commit it names eight slides in lesson 9 alone.
- **Multi-word `\text{}` lost its spaces in every equation image.** Lesson 2
  slide 32 projected `onlyiffisindependentofT`. `render_math` maps `\text` to
  mathtext's `\mathrm`, which sets the argument as one run; it now sets one
  `\mathrm` per word. Lesson 8's "chosen next" and lesson 10's "output size"
  were the other two. Inline maths had the mirror problem, a space the control
  word only needed as a terminator: `$\Delta G$` came out as "Δ G".
- **One slide carried bullets and a display equation together** - lesson 6's
  soft-margin formulation, the layout CLAUDE.md warns about. The build claimed
  to warn about it and only checked for text *after* an equation; it now checks
  for a bullet list beside one. Exactly one slide in the course tripped it.

Also found here, though it belongs to item 6: **three lessons still quoted 98 of
128 imputed rows**, which this review corrected to 94 in lesson 2. Lessons 8, 9
and 10 each repeat lesson 2's number in a carry-home list, and nothing checks
that a number quoted from another lesson still matches its source.

### Still open across the course

Written down 28 August 2026, after the notebook pass finished. Ordered by what
each is worth.

| # | What | Where it stands |
|---|---|---|
| 1 | **Check 2.1 — open every figure, confirm the caption describes *that* image.** The most productive check in this document, and the one no tool can do. | **Done for all ten lessons.** The faults cluster hard: five each in lessons 2 and 3, none of substance anywhere else. Lessons 4–8 and 10 needed only four legend or label placements; lesson 5 needed two captions that stated the textbook version of what the figure complicates; lesson 1's three faults were found and fixed in the August review and the fixes have held. |
| 2 | **Check 2.7 — look at the rendered deck.** | **Done for all ten.** Every deck rendered to one PNG per slide and read. Twenty overrunning slides, three equation-rendering faults and one bullets-beside-an-equation slide, all fixed; the section above records them. Two of the three now have an automated check that did not exist before, so the same faults cannot return silently. |
| 3 | **CLAUDE.md's "one number per lesson" table stops at lesson 3.** | **Done.** All ten rows filled, each with a figure verified during this review rather than chosen from memory. |
| 4 | **Two submission conventions.** Exercises 1–3 are due "at the start of Lesson N+1"; exercises 4–10 are due "23:59" on the same Friday. The dates chain correctly end to end — every "Set" equals the previous "Due", weekly from 25 September to 4 December — so only the *time* differs. | **Analysed, left for the instructor.** The two forms mean different deadlines: morning of the Friday against the end of it. Seven of ten use 23:59 and three use "start of Lesson N+1", and lessons 2's handout and exercise are internally consistent with the latter. Either choice is defensible and either changes what some students are told, so it is a teaching decision rather than a defect to fix silently. |
| 5 | **Deck density is nobody's decision.** 46 to 66 slides per lesson against roughly 80 lecture minutes: 35–50 slides an hour, where CLAUDE.md says 25–30. `verify_lesson.py` checks the plan sums to 180 minutes and nothing checks this ratio. | Not started; a teaching decision rather than a defect. |
| 6 | **Cross-lesson claims.** Each lesson makes checkable assertions about the others — 144 references in all. | **Done for the falsifiable subset.** Every reference naming a section number or quoting a figure was checked: four of them, three correct and one wrong (lesson 10's attribution of its dense baseline to lesson 9). The remaining 140 are prose pointers of the form "lesson 5 returns to this", which resolve but carry nothing to falsify. |


---

## 4. When a number moves

If re-running a notebook changes a number the material quotes, decide whether the
number is *stable* before reprinting it. A figure that depends on an ill-conditioned
solve will move again at the next image rebuild.

> **Lesson 3:** the largest coefficient of the degree-12 fit is deterministic inside
> the image and moved by a factor of 12,500 when the linear-algebra backend changed —
> the design matrix is full rank with condition number 4.2e9. The lesson's carry-home
> number was moved to one that had not budged.
