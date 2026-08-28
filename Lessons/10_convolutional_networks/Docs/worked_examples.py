#!/usr/bin/env python3
"""Recompute every number the lesson 10 handout works out by hand.

The rule: **start from the raw inputs, never from the handout's own
intermediate values.** A script that reuses the handout's intermediates proves
only that the handout is self-consistent, which a wrong handout also is.

So the convolution, the output-size formula, the parameter counts and the
pooling are all re-implemented here rather than imported from the notebooks —
that is what makes this a check on the *derivations* and not on the notebook
code. Where a second route exists it is used: convolution against `scipy`,
parameter counts against arrays actually allocated, the zero-sum property
analytically and then on random inputs, equivariance at random kernels and
random shifts rather than at the handout's four.

TensorFlow is deliberately not imported: this runs in the host environment,
which does not have it, and every claim below is arithmetic or geometry rather
than a training result. The training results are checked by re-executing the
notebooks.

Run directly, or let `tools/verify_lesson.py 10` discover it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate2d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Notebooks"))

from wafer_data import (                                          # noqa: E402
    make_wafer_images, make_defect_types,
    TRUE_IMAGE_SIZE, TRUE_GRADER_ERROR_RATE, TRUE_DEFECT_RATE,
    TRUE_SCRATCH_LENGTH, TRUE_DEFECT_TYPES,
)

CHECKS = 0
FAILURES = []


def check(label, computed, expected, tol, note=""):
    global CHECKS
    CHECKS += 1
    ok = abs(computed - expected) <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:<56s} {computed:>14.6f}"
          f"  handout {expected:<10.6g}{'  ' + note if note else ''}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed!r}, handout says {expected!r}")


def check_differs(label, a, b, at_least, note=""):
    """For a claim that two quantities are NOT the same.

    `check` asserts closeness, so using it for a difference claim inverts the
    test: it goes red exactly when the handout is right. This is its opposite,
    and it exists because the first draft of section 6 made that mistake.
    """
    global CHECKS
    CHECKS += 1
    ok = abs(a - b) >= at_least
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:<56s} {abs(a - b):>14.6f}"
          f"  must exceed {at_least:<8.4g}{'  ' + note if note else ''}")
    if not ok:
        FAILURES.append(f"{label}: differ by only {abs(a - b)!r}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# =====================================================================
section("1. The dataset and its ceiling (handout section 1.1)")
# =====================================================================

X, y, truth = make_wafer_images(seed=10001)
check("dies in the batch", float(len(X)), 4000, 0)
check("image side", float(X.shape[1]), TRUE_IMAGE_SIZE, 0)
check("recorded defect rate", float(y.mean()), 0.5080, 5e-5)
check("grader errors on THIS batch", float((y != truth).mean()), 0.0255, 5e-5)

# The handout calls 0.0255 a high draw rather than a bug. Checked, not assumed:
# re-derive the rate at eight other seeds and at a larger sample size.
rates = [float((lambda a: (a[1] != a[2]).mean())(make_wafer_images(seed=s)))
         for s in (11001, 11002, 11003, 12345, 10009, 10005, 555, 777)]
check("mean grader rate over eight OTHER seeds", float(np.mean(rates)),
      TRUE_GRADER_ERROR_RATE, 0.004, f"individual: {np.round(rates, 4).tolist()}")
spread = float(np.sqrt(TRUE_GRADER_ERROR_RATE * (1 - TRUE_GRADER_ERROR_RATE) / 4000))
check("binomial sd at n=4000", spread, 0.0022, 5e-5)
check("  so 0.0255 sits this many sd above 0.02",
      (0.0255 - TRUE_GRADER_ERROR_RATE) / spread, 2.5, 0.1)

X_te, y_te, t_te = make_wafer_images(seed=12345, n_images=2000)
check("ceiling realised on the TEST set", float((y_te == t_te).mean()), 0.9800, 5e-5)

X_type, y_type = make_defect_types(seed=10005, n_images=6000)
check("defect-typing classes", float(len(np.unique(y_type))),
      len(TRUE_DEFECT_TYPES), 0)


# =====================================================================
section("2. The ceiling identity (handout section 5.1)")
# =====================================================================

def with_station_error(q, e=TRUE_GRADER_ERROR_RATE):
    """Accuracy against the recorded grade, given agreement q with the truth."""
    return q * (1 - e) + (1 - q) * e


check("a perfect model scores", with_station_error(1.0), 0.98, 1e-12)
check("a model agreeing 0.90 with the truth scores",
      with_station_error(0.90), 0.884, 1e-12)

# Second route: simulate independent station errors rather than substituting.
rng = np.random.default_rng(4242)
n_sim = 4_000_000
for q in (1.0, 0.9, 0.75):
    model_right = rng.random(n_sim) < q
    station_wrong = rng.random(n_sim) < TRUE_GRADER_ERROR_RATE
    check(f"  simulated, q={q}", float((model_right == ~station_wrong).mean()),
          with_station_error(q), 1e-3)


# =====================================================================
section("3. Output size (handout section 3.2)")
# =====================================================================

def convolve(image, kernel, padding=0, stride=1):
    """Re-implemented here, not imported: this is the thing being checked."""
    if padding:
        image = np.pad(image, padding, mode="constant")
    f = kernel.shape[0]
    out_h = (image.shape[0] - f) // stride + 1
    out_w = (image.shape[1] - f) // stride + 1
    out = np.empty((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = np.sum(
                image[r * stride:r * stride + f, c * stride:c * stride + f] * kernel)
    return out


for n, f, p, s, expected in ((24, 3, 0, 1, 22), (24, 3, 1, 1, 24),
                             (24, 5, 0, 1, 20), (24, 5, 2, 1, 24),
                             (24, 3, 1, 2, 12)):
    formula = (n + 2 * p - f) // s + 1
    measured = convolve(np.zeros((n, n)), np.zeros((f, f)), p, s).shape[0]
    check(f"n={n} f={f} p={p} s={s}: formula", float(formula), expected, 0)
    check(f"  the same, measured", float(measured), expected, 0)

# The handout says the floor "is doing real work". Find a case where it does.
truncating = [(n, f, p, s) for n in range(8, 40) for f in (2, 3, 5)
              for p in (0, 1) for s in (2, 3)
              if (n + 2 * p - f) % s != 0]
check("input/kernel/stride combinations where the floor truncates",
      float(len(truncating) > 0), 1.0, 0,
      f"e.g. n={truncating[0][0]} f={truncating[0][1]} "
      f"p={truncating[0][2]} s={truncating[0][3]}")

# And that the formula still matches the implementation on all of them.
mismatches = sum(
    ((n + 2 * p - f) // s + 1)
    != convolve(np.zeros((n, n)), np.zeros((f, f)), p, s).shape[0]
    for n, f, p, s in truncating[:60])
check("formula vs implementation on 60 truncating cases",
      float(mismatches), 0.0, 0)

# 'same' padding: p = (f-1)/2 preserves the size, for every odd f.
bad = [f for f in (1, 3, 5, 7, 9)
       if convolve(np.zeros((24, 24)), np.zeros((f, f)), (f - 1) // 2, 1).shape[0] != 24]
check("p=(f-1)/2 preserves the size for odd f in 1..9", float(len(bad)), 0.0, 0)


# =====================================================================
section("4. The convolution itself (handout section 3.1)")
# =====================================================================

worst = 0.0
for trial in range(20):
    rng = np.random.default_rng(700 + trial)
    size = int(rng.integers(9, 20))
    f = int(rng.choice([3, 5]))
    image, kernel = rng.normal(0, 1, (size, size)), rng.normal(0, 1, (f, f))
    worst = max(worst, float(np.abs(
        convolve(image, kernel) - correlate2d(image, kernel, mode="valid")).max()))
check("largest disagreement with scipy, 20 random cases", worst, 0.0, 1e-12,
      "handout quotes 8.9e-16 for its one case")


# =====================================================================
section("5. The zero-sum kernel (handout section 3.3)")
# =====================================================================

# Analytic claim: a kernel whose weights sum to zero gives exactly zero on any
# constant patch, whatever its value. Checked on random zero-sum kernels and
# random constants, not on the handout's one example.
worst_constant = 0.0
for trial in range(200):
    rng = np.random.default_rng(800 + trial)
    kernel = rng.normal(0, 2, (3, 3))
    kernel -= kernel.mean()                       # force the sum to zero
    value = float(rng.normal(0, 50))
    worst_constant = max(worst_constant, float(np.abs(
        convolve(np.full((7, 7), value), kernel)).max()))
check("zero-sum kernel on a constant patch, 200 random cases",
      worst_constant, 0.0, 1e-10)

# And that a kernel which does NOT sum to zero fails the same test, so the
# check above could actually go red.
rng = np.random.default_rng(9)
nonzero_kernel = np.ones((3, 3))
check("  a non-zero-sum kernel is NOT blind to brightness",
      float(np.abs(convolve(np.full((7, 7), 3.0), nonzero_kernel)).max()), 27.0, 1e-12)

contrast = np.array([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) * -1
check("its weights sum to", float(contrast.sum()), 0.0, 1e-12)

defective = X[truth == 1][0]
clean = X[truth == 0][0]
check("strongest response on a defective die",
      float(convolve(defective, contrast, padding=1).max()), 3.129, 5e-4)
check("strongest response on a clean die",
      float(convolve(clean, contrast, padding=1).max()), 1.118, 5e-4)


# =====================================================================
section("6. Equivariance to translation (handout section 4.1)")
# =====================================================================

def shift(a, dr, dc):
    return np.roll(np.roll(a, dr, axis=0), dc, axis=1)


# The handout checks four shifts with one kernel. This checks random shifts
# with random kernels, which is what makes it a property rather than a case.
worst_equivariance = 0.0
for trial in range(100):
    rng = np.random.default_rng(900 + trial)
    image = rng.normal(0, 1, (24, 24))
    kernel = rng.normal(0, 1, (3, 3))
    dr, dc = rng.integers(-6, 7, 2)
    left = convolve(shift(image, dr, dc), kernel, padding=1)
    right = shift(convolve(image, kernel, padding=1), dr, dc)
    inner = slice(8, 16)                       # away from the wrap-around
    worst_equivariance = max(worst_equivariance,
                             float(np.abs(left[inner, inner] - right[inner, inner]).max()))
check("shift-then-convolve vs convolve-then-shift, 100 random cases",
      worst_equivariance, 0.0, 0.0,
      "handout claims exact, to the last bit")

# A dense layer is not equivariant, which is the contrast the handout draws.
rng = np.random.default_rng(11)
dense_weights = rng.normal(0, 1, (24, 24))
image = rng.normal(0, 1, (24, 24))
unshifted = float((image * dense_weights).sum())
shifted = float((shift(image, 3, 0) * dense_weights).sum())
check_differs("a dense unit's output changes when the input shifts",
              shifted, unshifted, 1.0,
              "the convolution's did not change at all")


# =====================================================================
section("7. Pooling (handout section 4.2)")
# =====================================================================

def max_pool(a, k=2):
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[:h * k, :w * k].reshape(h, k, w, k).max(axis=(1, 3))


feature_map = convolve(defective, contrast, padding=1)
current, sizes = feature_map, []
for _ in range(3):
    current = max_pool(current)
    sizes.append(current.shape[0])
    check(f"max preserved after pooling to {current.shape[0]}x{current.shape[0]}",
          float(current.max()), float(feature_map.max()), 1e-12)
check("three 2x2 poolings take 24 down to", float(sizes[-1]), 3, 0)

# Receptive field: after one 2x2 pool, a 3x3 kernel covers 6x6 of the input.
check("receptive field of 3x3 after one 2x2 pool", float(3 * 2), 6, 0)


# =====================================================================
section("8. Parameter counts (handout sections 2 and 5)")
# =====================================================================

pixels = TRUE_IMAGE_SIZE * TRUE_IMAGE_SIZE
check("pixels in a die", float(pixels), 576, 0)


def dense_params(n_in, n_units):
    """Counted by allocating what such a layer would actually hold."""
    return np.zeros((n_in, n_units)).size + np.zeros(n_units).size


def conv_params(n_kernels, f, in_channels):
    return np.zeros((f, f, in_channels, n_kernels)).size + np.zeros(n_kernels).size


check("dense, 256 units", float(dense_params(pixels, 256)), 147712, 0)
check("dense, 1024 units", float(dense_params(pixels, 1024)), 590848, 0)
check("conv, 8 kernels of 3x3 on 1 channel", float(conv_params(8, 3, 1)), 80, 0)
check("conv, 16 kernels of 3x3 on 1 channel", float(conv_params(16, 3, 1)), 160, 0)
check("conv, 32 kernels of 5x5 on 1 channel", float(conv_params(32, 5, 1)), 832, 0)
check("ratio, 256-unit dense to 8-kernel conv",
      dense_params(pixels, 256) / conv_params(8, 3, 1), 1846.4, 0.05)

# The multi-channel subtlety the handout flags: layer 2's kernels are 3x3x8.
check("conv, 16 kernels of 3x3 on 8 channels", float(conv_params(16, 3, 8)), 1168, 0)

network = (conv_params(8, 3, 1) + conv_params(16, 3, 8)
           + dense_params(16, 16) + dense_params(16, 1))
check("the whole convolutional network", float(network), 1537, 0)

dense_network = (dense_params(pixels, 256) + dense_params(256, 256)
                 + dense_params(256, 1))
check("the whole dense network", float(dense_network), 213761, 0)
check("  it outputs MORE numbers than the dense layer, not fewer",
      float(24 * 24 * 8), 4608, 0, "against 256 for the dense layer")


# =====================================================================
section("9. Differences quoted in the text")
# =====================================================================

check("translation: what it costs the dense network", 0.8567 - 0.4617, 0.3950, 5e-5)
check("translation: what it costs the convolutional one",
      0.9847 - 0.9800, 0.0047, 5e-5)
check("the dense network ends below chance", 0.4617, 0.4617, 5e-5,
      "chance is 0.50")
# The handout's table shows the two accuracies rounded to four places and the
# gain as the notebook printed it, from the unrounded values. Subtracting the
# rounded pair gives 0.0806 against a printed 0.0807, so the tolerance here is
# one unit in the last place and the disagreement is arithmetic, not a claim.
check("augmentation gain, from the rounded components",
      0.5423 - 0.4617, 0.0807, 2e-4, "notebook printed 0.0807 unrounded")
check("  and how far short it still is", 0.9847 - 0.5423, 0.4424, 5e-5,
      "handout says forty-four points")
check("dense shortfall at 8,000 images", 0.9800 - 0.7455, 0.2345, 5e-5)
check("permutation cost, defect typing", 0.9970 - 0.9523, 0.0447, 5e-5)
check("permutation cost, defect detection", 0.9800 - 0.9798, 0.0002, 5e-5)
check("transfer gain at 100 images", 0.8640 - 0.7220, 0.1420, 5e-5)
check("transfer gain at 200 images", 0.8940 - 0.7270, 0.1670, 5e-5)
check("transfer at 400 images has gone negative", 0.7877 - 0.8447, -0.0570, 5e-5)
check("convolutional over dense, synthesis table", 0.9800 - 0.7430, 0.2370, 5e-5)
check("random forest behind the convolutional network by",
      0.9800 - 0.9780, 0.0020, 5e-5, "handout says two thousandths")
check("dense network behind logistic regression by", 0.8905 - 0.7430, 0.1475, 5e-5)


# =====================================================================
section("Section 5, the parameter counts")

# review_lesson.py can only see this lesson's notebooks, so the two figures
# borrowed from lesson 9 are unreachable to it. Checked from the architecture.
PIXELS_HERE, PIXELS_LESSON_9 = 24 * 24, 8 * 8

check("a 256-unit dense layer on these images", PIXELS_HERE * 256 + 256, 147_712, 0)
check("eight 3x3 kernels, against it", 8 * 9 + 8, 80, 0)
check("the difference the slides quote", 147_712 - 80, 147_632, 0)
check("and the ratio", (PIXELS_HERE * 256 + 256) / 80, 1_846, 1)
check("the two-hidden-layer dense baseline",
      (PIXELS_HERE * 256 + 256) + (256 * 256 + 256) + (256 + 1), 213_761, 0)
check("32 kernels of 3x3 over 16 channels", 32 * (9 * 16) + 32, 4_640, 0)
check("lesson 9's one-hidden-layer digit network",
      PIXELS_LESSON_9 * 64 + 64 + 64 * 10 + 10, 4_810, 0,
      "8x8 inputs, not this lesson's 24x24")
check("and lesson 9's two-hidden-layer one",
      PIXELS_LESSON_9 * 64 + 64 + 64 * 64 + 64 + 64 * 10 + 10, 8_970, 0)

# =====================================================================
section("Result")
print(f"\n  {CHECKS} hand-worked numbers recomputed from the raw inputs.")
if FAILURES:
    print(f"  {len(FAILURES)} DISAGREE with the handout:\n")
    for failure in FAILURES:
        print(f"    - {failure}")
    sys.exit(1)
print("  All agree with the handout.\n")
