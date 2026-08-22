"""Data for Lesson 10 — Convolutional networks.

Lesson 9 closed on a promise: everything in it treated the 64 pixels of a
digit as 64 unrelated numbers, and permuting them consistently across the
dataset would change nothing. This lesson cashes that in, so it needs images
where **position matters** — and where the same thing can appear in more than
one place.

Meridian Instruments, whose sensors ran through lesson 9, inspects the silicon
wafers those sensors are cut from. Each wafer is photographed and a small
region is graded. A die is rejected when it carries a defect, and a defect is
a *local* pattern: a scratch a few pixels long, or a bright particle. The
pattern is the same wherever it lands, which is precisely the structure a
convolution is built to exploit and a dense layer has to learn separately at
every position.

Three generators:

1. ``make_wafer_images`` — the main set. Defects anywhere, or restricted to
   the top or bottom half, which is what makes the translation experiment
   possible: train where the defects have been seen, test where they have not.
2. ``make_rare_defect`` — a second, rarer defect class with few labelled
   examples, for transfer learning. Nothing is downloaded: the "pre-trained"
   features come from a network trained earlier in the same notebook.
3. ``load_digit_images`` — lesson 9's digits, unchanged, plus a fixed pixel
   permutation for the experiment that opens this lesson.

Every ``TRUE_*`` constant below records how the labels were made. The
inspection station is imperfect at a published rate, which puts a ceiling on
every score in this lesson.

No network access, no API keys.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

TRUE_IMAGE_SIZE = 24                 #: images are 24 x 24, single channel
TRUE_N_IMAGES = 4000                 #: dies in the main batch
TRUE_DEFECT_RATE = 0.5               #: half of them carry a defect

#: The grader mislabels this fraction of dies, so 0.98 is the best accuracy
#: anything in this lesson can reach against the recorded labels.
TRUE_GRADER_ERROR_RATE = 0.02
TRUE_BAYES_ACCURACY = 1.0 - TRUE_GRADER_ERROR_RATE

# --- the wafer surface the defect sits on -----------------------------------
#: Film thickness varies smoothly across a die, which is why the background is
#: not flat: a defect detector that keys on absolute brightness will fail.
TRUE_BACKGROUND_LEVEL = 0.50
TRUE_BACKGROUND_SWING = 0.14         #: amplitude of the smooth variation
TRUE_SENSOR_NOISE = 0.045            #: per-pixel camera noise

# --- defect appearance ------------------------------------------------------
TRUE_SCRATCH_LENGTH = 7              #: pixels, the long axis of a scratch
TRUE_SCRATCH_CONTRAST = -0.34        #: scratches are darker than the surface
TRUE_PARTICLE_SIGMA = 0.95           #: pixels, the blob's spread
TRUE_PARTICLE_CONTRAST = 0.46        #: particles are brighter

#: A contamination cluster: several specks, each smaller and fainter than a
#: lone particle, but of the same polarity — bright on a darker surface. The
#: contrast is set low deliberately: at 0.34 a network learns this defect from
#: a hundred examples unaided and transfer learning has nothing left to
#: contribute, which makes for a demonstration that demonstrates nothing.
TRUE_CLUSTER_SPECKS = 3
TRUE_CLUSTER_SIGMA = 0.75
TRUE_CLUSTER_CONTRAST = 0.24

#: A defect centre never lands closer than this to an edge, so that the whole
#: motif is visible and a model cannot succeed by watching the border alone.
TRUE_DEFECT_MARGIN = 5

#: Row bands used by the translation experiment, as (low, high) centre rows.
TRUE_TOP_BAND = (5, 11)
TRUE_BOTTOM_BAND = (13, 19)


def _background(rng, n, size):
    """Smooth film-thickness variation plus camera noise."""
    grid = np.arange(size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    out = np.empty((n, size, size))
    for i in range(n):
        # Two low-frequency components with random phase and orientation.
        image = np.zeros((size, size))
        for _ in range(2):
            freq = rng.uniform(0.06, 0.16)
            angle = rng.uniform(0, np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            projection = xx * np.cos(angle) + yy * np.sin(angle)
            image += np.sin(2 * np.pi * freq * projection + phase)
        out[i] = TRUE_BACKGROUND_LEVEL + TRUE_BACKGROUND_SWING * image / 2
    out += rng.normal(0, TRUE_SENSOR_NOISE, out.shape)
    return out


def _draw_scratch(image, rng, row, col):
    """A short dark line segment centred on (row, col)."""
    angle = rng.uniform(0, np.pi)
    half = (TRUE_SCRATCH_LENGTH - 1) / 2
    steps = np.linspace(-half, half, TRUE_SCRATCH_LENGTH * 3)
    for s in steps:
        r = int(round(row + s * np.sin(angle)))
        c = int(round(col + s * np.cos(angle)))
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            image[r, c] += TRUE_SCRATCH_CONTRAST
    return image


def _draw_particle(image, rng, row, col):
    """A small bright Gaussian blob centred on (row, col)."""
    size = image.shape[0]
    grid = np.arange(size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    squared = (yy - row) ** 2 + (xx - col) ** 2
    image += TRUE_PARTICLE_CONTRAST * np.exp(
        -squared / (2 * TRUE_PARTICLE_SIGMA ** 2))
    return image


def _draw_cluster(image, rng, row, col):
    """Three small bright specks close together — a contamination cluster.

    Deliberately built from the same material as a particle but arranged
    differently: it is the defect type used for transfer, and the point is
    that a network which has seen particles has already learned most of what
    it needs to see this.
    """
    for _ in range(TRUE_CLUSTER_SPECKS):
        dr, dc = rng.integers(-2, 3, 2)
        size = image.shape[0]
        grid = np.arange(size)
        yy, xx = np.meshgrid(grid, grid, indexing="ij")
        squared = (yy - (row + dr)) ** 2 + (xx - (col + dc)) ** 2
        image += TRUE_CLUSTER_CONTRAST * np.exp(
            -squared / (2 * TRUE_CLUSTER_SIGMA ** 2))
    return image


PAINTERS = {"scratch": _draw_scratch, "particle": _draw_particle,
            "cluster": _draw_cluster}


def _sample_centre(rng, n, size, band):
    """Defect centres, either anywhere legal or inside a row band."""
    low, high = TRUE_DEFECT_MARGIN, size - TRUE_DEFECT_MARGIN
    cols = rng.integers(low, high, n)
    if band is None:
        rows = rng.integers(low, high, n)
    else:
        rows = rng.integers(band[0], band[1], n)
    return rows, cols


def make_wafer_images(seed=10001, n_images=TRUE_N_IMAGES, band=None,
                      defect="scratch"):
    """Inspection images of ``n_images`` dies, half of them defective.

    ``band`` restricts where a defect may appear: ``None`` for anywhere,
    ``TRUE_TOP_BAND`` or ``TRUE_BOTTOM_BAND`` for one half. Training on one
    band and testing on the other is the experiment that separates a dense
    network from a convolutional one, because it asks the model about a
    position it has never seen a defect in.

    ``defect`` is ``"scratch"``, ``"particle"`` or ``"cluster"``.

    Returns ``(X, y, truth)`` with X of shape (n, 24, 24) in roughly [0, 1],
    y the *recorded* grade (0 clean, 1 defective) and truth the noise-free
    grade. A model is only ever shown X and y.
    """
    rng = np.random.default_rng(seed)
    size = TRUE_IMAGE_SIZE
    X = _background(rng, n_images, size)

    truth = (rng.random(n_images) < TRUE_DEFECT_RATE).astype(int)
    rows, cols = _sample_centre(rng, n_images, size, band)
    painter = PAINTERS[defect]
    for i in np.where(truth == 1)[0]:
        painter(X[i], rng, rows[i], cols[i])

    flip = rng.random(n_images) < TRUE_GRADER_ERROR_RATE
    recorded = np.where(flip, 1 - truth, truth)
    return np.clip(X, 0.0, 1.0), recorded, truth


#: Class codes for the defect-typing task used to pre-train.
TRUE_DEFECT_TYPES = ("clean", "scratch", "particle")


def make_defect_types(seed=10005, n_images=6000):
    """The richer task used for pre-training: which of three types is this?

    Transfer learning works when the source task forces the network to learn
    features the target task also needs. Grading a die pass/fail on one defect
    teaches one detector. Naming *which* of several defects is present forces
    the early layers to describe local structure in general — dark lines and
    bright blobs both — and those are the features worth carrying over.

    Returns ``(X, y)`` with y in 0, 1, 2 indexing ``TRUE_DEFECT_TYPES``.
    """
    rng = np.random.default_rng(seed)
    size = TRUE_IMAGE_SIZE
    X = _background(rng, n_images, size)
    y = rng.integers(0, len(TRUE_DEFECT_TYPES), n_images)
    rows, cols = _sample_centre(rng, n_images, size, None)
    for i in np.where(y > 0)[0]:
        PAINTERS[TRUE_DEFECT_TYPES[y[i]]](X[i], rng, rows[i], cols[i])
    return np.clip(X, 0.0, 1.0), y


def make_rare_defect(seed=10002, n_images=300):
    """A defect type nobody has many pictures of yet, for transfer learning.

    A contamination cluster: a *new* type, absent from the pre-training task,
    but built of the same bright specks a particle is made of. That is the
    situation transfer is for — a new defect appears on the line on Monday and
    there are forty labelled pictures of it by Friday.
    """
    return make_wafer_images(seed=seed, n_images=n_images, defect="cluster")


# ---------------------------------------------------------------------------
# Lesson 9's digits, and the permutation that opens this lesson
# ---------------------------------------------------------------------------

TRUE_N_CLASSES = 10
DIGIT_IMAGE_SHAPE = (8, 8)
DIGIT_MAX_INTENSITY = 16.0


def load_digit_images(seed=9003, val_size=0.2, test_size=0.2):
    """Exactly lesson 9's split of the scikit-learn 8x8 digits.

    Reproduced here with the same seed and the same proportions so that this
    lesson's convolutional scores can be read directly against lesson 9's
    dense ones.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data / DIGIT_MAX_INTENSITY
    y = digits.target

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1.0 - test_size),
        stratify=y_rest, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


def pixel_permutation(seed=10003, n_pixels=64):
    """One fixed shuffling of the pixel positions, applied to every image.

    The point of lesson 9's closing paragraph: a dense network cannot tell
    this has been done, because it never used the arrangement in the first
    place. A convolution can, because the arrangement is all it uses.
    """
    return np.random.default_rng(seed).permutation(n_pixels)
