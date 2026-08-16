# Peer review form

**Project reviewed:** ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
**Reviewer:** ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
**Date:** ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­

Your job is **not** to judge whether the model is good. It is to find the places where
the conclusion might not hold. A review that finds a real flaw is more useful to your
colleague than one that says the work is fine.

Be specific and be kind. Point at cells and lines.

---

## 1. Does it run?

- [ ] The notebook runs top to bottom in the course container without errors
- [ ] A second run produces the same numbers
- [ ] No package outside the course image is needed

If it failed, where?

> 

---

## 2. Is the evaluation honest?

This is the heart of the review. Work through each question.

**Was the evaluation designed before the modelling, or fitted around the result?**

> 

**Could anything from the test set have influenced a decision?** Check scaling,
imputation, feature selection, encoding of categories, and hyperparameter tuning.
Look specifically for a transformer fitted on the full dataset before splitting.

> 

**Does the splitting scheme suit the data?** If there is a temporal ordering, a group
structure, or repeated measurements of the same entity, a random split will
overestimate performance. Is that handled?

> 

**Is the metric the right one for the problem?** Accuracy on imbalanced data, R² on a
skewed target, a single threshold with no justification — flag any of these.

> 

**Is the comparison between models fair?** Same data, same preprocessing, same
tuning effort for each?

> 

---

## 3. Is the reasoning sound?

**Is there a baseline, and is the improvement over it meaningful?**

> 

**Do the stated conclusions follow from the evidence shown?** Look for claims the
numbers do not support.

> 

**Is the uncertainty acknowledged?** A single number with no sense of its variability
across folds or seeds is a weak result.

> 

---

## 4. Can you follow it?

**Could you reproduce this study from the write-up alone?**

> 

**Which part was hardest to follow, and why?**

> 

---

## 5. The most useful thing you can say

**One change that would most improve this project:**

> 

**One thing done well that you intend to steal for your own:**

> 

---

## Reviewer checklist

- [ ] I read the whole notebook, not only the conclusions
- [ ] I checked the preprocessing for leakage specifically
- [ ] My comments point at specific cells
- [ ] I said at least one concrete thing that would improve the work
