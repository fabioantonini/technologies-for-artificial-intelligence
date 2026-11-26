"""The checkpoints of the workshop notebook.

Each ``stepN`` function looks at what you computed, prints a tick or a cross
with a hint, and records the result for ``summary()``. The expected values are
recomputed here from the same data, with the same split and the same folds, so
a check compares your work with an honest reference rather than with a number
typed by hand.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from delivery_data import load_deliveries

warnings.filterwarnings("ignore", category=UserWarning)

_X, _y = load_deliveries()
_Xtr, _Xte, _ytr, _yte = train_test_split(_X, _y, test_size=0.25, stratify=_y,
                                          random_state=0)
_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
_STRAIGHT = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000))
_CURVED = make_pipeline(SimpleImputer(strategy="median"),
                        PolynomialFeatures(degree=2, include_bias=False),
                        StandardScaler(), LogisticRegression(max_iter=5000))
_STRAIGHT_CV = cross_val_score(_STRAIGHT, _Xtr, _ytr, cv=_FOLDS)
_CURVED_CV = cross_val_score(_CURVED, _Xtr, _ytr, cv=_FOLDS)
_STRAIGHT_TEST = _STRAIGHT.fit(_Xtr, _ytr).score(_Xte, _yte)
_CURVED_TEST = _CURVED.fit(_Xtr, _ytr).score(_Xte, _yte)

_RESULTS: dict[int, bool] = {}


def _report(step: int, ok: bool, good: str, bad: str) -> None:
    _RESULTS[step] = ok
    print(f"{'✓' if ok else '✗'} Step {step}: {good if ok else bad}")


def step1(baseline) -> None:
    """The accuracy of always answering 'on time'."""
    expected = 1 - _y.mean()
    ok = abs(float(baseline) - expected) < 0.005
    _report(1, ok,
            f"the baseline is {expected:.3f}. Any model has to beat that to be worth anything.",
            "not yet. The baseline is the share of the most common class: the fraction "
            "of parcels that arrived on time. Try 1 - y.mean().")


def step2(X_train, X_test, y_train, y_test) -> None:
    """A stratified split, one quarter for testing."""
    sizes = (len(X_train), len(X_test)) == (900, 300)
    stratified = abs(y_train.mean() - y_test.mean()) < 0.002
    _report(2, sizes and stratified,
            f"900 parcels to fit, 300 to test, and {y_train.mean():.1%} against "
            f"{y_test.mean():.1%} late in the two parts.",
            ("the sizes should be 900 and 300: test_size=0.25. " if not sizes else "")
            + ("The late share differs between the two parts: pass stratify=y."
               if not stratified else ""))


def step3(model, X_train) -> None:
    """Imputer and scaler inside the pipeline, before the model."""
    if not isinstance(model, Pipeline):
        _report(3, False, "", "model should be built with make_pipeline(...).")
        return
    kinds = [type(step).__name__ for _, step in model.steps]
    inside = kinds == ["SimpleImputer", "StandardScaler", "LogisticRegression"]
    untouched = X_train["rain_mm"].isna().any()
    _report(3, inside and untouched,
            "imputer, scaler and model in one pipeline, and X_train still has its gaps: "
            "nothing has been learned from the data outside the pipeline.",
            ("the pipeline should be SimpleImputer, StandardScaler, LogisticRegression, "
             f"in that order; it is {kinds}. " if not inside else "")
            + ("X_train has no gaps left: the imputing must happen inside the pipeline, "
               "not on the data beforehand." if not untouched else ""))


def step4(scores) -> None:
    """Cross-validation on the training set only."""
    scores = np.asarray(scores)
    ok = scores.shape == (5,) and np.allclose(scores, _STRAIGHT_CV)
    _report(4, ok,
            f"{scores.mean():.3f} ± {scores.std():.3f} on five folds of the training set, "
            f"against a baseline of {1 - _y.mean():.3f}.",
            "the scores differ from cross-validation on the training set with the given "
            "folds. Pass X_train and y_train: the test set must not take part.")


def step5(curved_scores) -> None:
    """The model with squared terms, on the same folds."""
    curved_scores = np.asarray(curved_scores)
    ok = curved_scores.shape == (5,) and np.allclose(curved_scores, _CURVED_CV)
    _report(5, ok,
            f"{curved_scores.mean():.3f} ± {curved_scores.std():.3f} with squared terms, "
            f"against {_STRAIGHT_CV.mean():.3f} ± {_STRAIGHT_CV.std():.3f} without, on "
            "the same five folds.",
            "not the expected scores: use degree=2, the same folds and X_train, y_train.")


def step6(test_accuracy) -> None:
    """One look at the test set, with the model that won step 5."""
    a = float(test_accuracy)
    if abs(a - _CURVED_TEST) < 1e-9:
        cv = _CURVED_CV
        _report(6, True,
                f"test accuracy {a:.3f}, cross-validation {cv.mean():.3f} ± {cv.std():.3f}: "
                "the two agree, so the cross-validation told the truth.", "")
    elif abs(a - _STRAIGHT_TEST) < 1e-9:
        _report(6, False, "",
                f"that is the straight model, {a:.3f}. Step 5 said the curved one is better: "
                "choose with the cross-validation, then test once.")
    else:
        _report(6, False, "",
                "not the expected number: fit the chosen model on X_train, y_train, then "
                "score it once on X_test, y_test.")


def summary() -> None:
    """Every step, ticked or not."""
    for step in range(1, 7):
        mark = {True: "✓", False: "✗"}.get(_RESULTS.get(step), "·")
        print(f"  {mark}  step {step}")
    print(f"\n{sum(_RESULTS.values())} of 6 steps passed.")
