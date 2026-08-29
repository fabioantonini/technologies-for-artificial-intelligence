# Exam

Two stages on the same day: a **written paper**, closed book, two hours; then a
**discussion** of one of the ten weekly exercises, drawn at random.

The exam tests what the project cannot: whether you understand *why* the methods work.
The paper draws on the **handouts**, including the derivations. If you have only run
the notebooks, you will not pass.

---

## The written paper

| Part | Questions | Marks | What it tests |
|---|---|---|---|
| **A — Foundations** | 8 short answers | 24 | Definitions, statements, when a method applies |
| **B — Derivations** | 2 of 3 offered | 30 | Reproduce and reason about a derivation from the handouts |
| **C — Method selection** | 2 scenarios | 26 | Choose an approach for a described problem and justify it |
| **D — Diagnosis** | 1 case | 20 | Given results, say what went wrong and how to check |

Part D always presents a flawed study. Finding the flaw is a course learning outcome,
so it is never worth fewer than a fifth of the marks.

---

## The discussion

One of the ten weekly exercises is drawn, and you talk through **your own notebook**
for it. **Bring all ten**, on a laptop or printed; an exercise you cannot produce is
an exercise you did not do.

It is a conversation about work you have already done, not a second examination of the
theory. Expect to be asked:

- what you actually measured, and why that quantity rather than another;
- which decision in the notebook you were least sure of, and what you did about it;
- what result would have told you the whole thing was wrong.

There are no marks for the score you obtained. There are marks for knowing why you
obtained it, which is the same standard the exercises are written to and the same one
the project is marked against.

Its weight beside the written paper is confirmed with the degree programme; see
[`../../Course/SYLLABUS.md`](../../Course/SYLLABUS.md).

---

## What you are expected to be able to derive

From the handouts, unaided:

| Lesson | Derivation |
|---|---|
| 3 | Normal equation from the least-squares objective; the gradient descent update; Ridge as constrained optimisation |
| 4 | The logistic model's cross-entropy from maximum likelihood; **why squared error is a poor choice for classification** |
| 5 | The bias-variance decomposition of expected prediction error |
| 6 | The SVM margin as an optimisation problem; what the kernel trick avoids computing |
| 7 | Entropy and information gain for a split; boosting as gradient descent in function space |
| 8 | PCA as an eigenvalue problem; the k-means objective and why the algorithm converges |
| 9 | Backpropagation for a two-layer network; why non-linear activations are necessary |

You are not asked to memorise library APIs. You may be asked to write a few lines of
pseudocode.

---

## Materials

- [`question_bank.md`](question_bank.md) — questions by topic, built up as the course runs
- `sample_paper_1.md`, `sample_paper_2.md` — full papers with solutions, published
  after Lesson 8

---

## How to prepare

Keep the ten exercise notebooks somewhere you can find them, and re-read your own
reasoning before the day rather than rewriting the code — the discussion is about the
decisions, and those live in the markdown cells.

For the paper: work through the derivations with a pen rather than reading them. Then take the quiz
for each lesson: the quizzes deliberately include questions that require reasoning
about a derivation rather than recalling a fact, and those are the closest thing to
Part B you will find before the sample papers appear.
