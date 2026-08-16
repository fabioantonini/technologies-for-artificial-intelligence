# Written exam

Closed book, two hours.

The exam tests what the project cannot: whether you understand *why* the methods work.
It draws on the **handouts**, including the derivations. If you have only run the
notebooks, you will not pass.

---

## Structure

| Part | Questions | Marks | What it tests |
|---|---|---|---|
| **A — Foundations** | 8 short answers | 24 | Definitions, statements, when a method applies |
| **B — Derivations** | 2 of 3 offered | 30 | Reproduce and reason about a derivation from the handouts |
| **C — Method selection** | 2 scenarios | 26 | Choose an approach for a described problem and justify it |
| **D — Diagnosis** | 1 case | 20 | Given results, say what went wrong and how to check |

Part D always presents a flawed study. Finding the flaw is a course learning outcome,
so it is never worth fewer than a fifth of the marks.

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

Work through the derivations with a pen rather than reading them. Then take the quiz
for each lesson: the quizzes deliberately include questions that require reasoning
about a derivation rather than recalling a fact, and those are the closest thing to
Part B you will find before the sample papers appear.
