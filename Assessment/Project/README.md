# Final project

An end-to-end machine learning study on a dataset you have not seen in class,
written up so that another student could reproduce it.

Presented at **Lesson 1** so you can think about it while the course runs. Topic
confirmed by **Lesson 4**. Submitted after Lesson 10.

---

## What you have to do

Take one of the datasets below, or propose your own, and carry out a complete study:

1. **Frame the problem.** What is predicted, from what, and why would anyone want it?
   State the target, the features, and what a useful result would look like.
2. **Explore and prepare the data.** Missing values, outliers, encoding, scaling.
   Document every decision and its justification.
3. **Establish a baseline.** The simplest thing that could work — the majority class,
   the mean, a single feature. Everything later is measured against this.
4. **Design the evaluation before you model.** Choose your split or cross-validation
   scheme, and your metric, and justify both *given the problem*. Write this down
   before you fit anything serious.
5. **Compare at least three model families** from the course, tuned honestly.
6. **Diagnose.** Learning curves, error analysis, where and why the best model fails.
7. **Interpret.** What drives the predictions? What would you tell someone who had to
   act on this?
8. **State the limitations.** What would you need to trust this in production? What
   could go wrong, and who would it affect?

**Deliberately not required:** a state-of-the-art result. A careful study that
concludes "this dataset does not support a useful model, and here is the evidence" is
a complete and successful project.

---

## Datasets

Pick one. All are public, load in seconds, and are small enough to run on a laptop.

| Dataset | Task | Why it is interesting | Source |
|---|---|---|---|
| **Bike sharing demand** | Regression | Strong temporal structure — a trap for naive random splitting | UCI, id 275 |
| **Adult census income** | Binary classification | Class imbalance plus genuine fairness questions across protected attributes | UCI, id 2 |
| **Online shoppers intention** | Binary classification | Heavily imbalanced; accuracy is actively misleading here | UCI, id 468 |
| **Wine quality** | Regression or ordinal | Small, correlated features, ambiguous target definition | UCI, id 186 |
| **Steel plates faults** | Multiclass | Seven classes, unevenly represented, industrial context | UCI, id 198 |

Load any of them with:

```python
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=..., as_frame=True)
X, y = data.data, data.target
```

**Proposing your own dataset** is welcome and must be agreed by Lesson 4. It has to be
public, licensed for teaching use, and non-trivial — no Titanic, no Iris, no MNIST.

---

## What to hand in

A folder named `{surname}_{studentid}` containing:

| File | Contents |
|---|---|
| `report.ipynb` | The study: narrative and code together, running top to bottom in the course container |
| `README.md` | Half a page: the question, what you did, what you found, how to run it |

Constraints:

- **Runs in the course Docker image with no extra installs.** If it needs a package
  we do not ship, it does not run.
- **Reproducible**: seeds fixed, and a second execution gives the same numbers.
- Outputs left in place, so it can be read without being executed.
- Roughly 2,000-3,000 words of narrative. Prose that explains decisions, not comments
  restating the code.

---

## Assessment criteria

| Criterion | Weight | What earns marks |
|---|---|---|
| **Methodological soundness** | 35% | Evaluation designed before modelling; no leakage; the split scheme suits the data; honest treatment of uncertainty |
| **Technical execution** | 25% | Correct implementation, sensible preprocessing, fair comparison between models |
| **Diagnosis and interpretation** | 20% | Understands *why* the model behaves as it does; error analysis goes beyond a single metric |
| **Communication** | 15% | A reader can follow the reasoning and reproduce the work |
| **Critical judgement** | 5% | Limitations stated honestly, including what would make the result untrustworthy |

Note what is absent: **there are no marks for accuracy**. Two students on the same
dataset with different scores can both receive full marks, and the higher score can be
the lower mark if it was obtained by contaminating the evaluation.

### Automatic deductions

| Problem | Effect |
|---|---|
| Test set used for any fitting decision — scaling, feature selection, tuning | Methodology capped at 40% |
| Results not reproducible on re-run | Execution capped at 50% |
| Notebook does not run in the course image | Returned unmarked for resubmission |

---

## Peer review

Before submission, each project is read by two other students, and each student reads
two projects. Use [`peer_review_form.md`](peer_review_form.md).

This is part of the assessment, not a formality: reviewing someone else's evaluation
scheme is one of the fastest ways to learn to see the flaws in your own. The reviews
you write are assessed alongside the project you submit.
