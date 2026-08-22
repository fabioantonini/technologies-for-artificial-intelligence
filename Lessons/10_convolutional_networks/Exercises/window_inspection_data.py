"""Two inspection stations on one coating line, for exercise 10.

Meridian Instruments — the fictional maker of optical distance sensors whose
wafers ran through lesson 10 — also coats the glass **windows** those sensors
look out through. A window is a disc of fused silica, anti-reflection coated
on both faces, pressed into a metal mount. After coating, every window is
photographed under a bright-field microscope and the photograph is cropped to
a 24 x 24 pixel patch covering the whole of the glass.

Two different questions are asked of that one photograph, at two stations on
the same line. This module gives you one batch from each, deliberately matched
so that almost nothing differs between them:

===========================  ==============  ==============
                             coating grade   aperture grade
===========================  ==============  ==============
images in the batch                   3,000           3,000
image size                       24 x 24 px      24 x 24 px
reject rate                          ~0.50           ~0.50
grades recorded wrongly                 3 %             3 %
===========================  ==============  ==============

Same size, same images-per-batch, same picture, same base rate, same label
noise, the same blemish drawn by the same code. The one thing that is *not*
the same is **what the station is asking about the blemish**, and that is the
whole exercise.

``load_coating_grade``   station 1, goods-inward: is this window blemished at
                         all?
``load_aperture_grade``  station 2, salvage: the window is blemished — can it
                         still be shipped?

Neither loader touches the network. Both generate their batch locally with
NumPy and return the same arrays on any machine.


What a window photograph looks like
-----------------------------------

The anti-reflection coating is a few hundred nanometres thick and its
thickness varies smoothly across the disc, so the patch is never flat: it
carries a slow, low-frequency swell of brightness with a random orientation
and phase, plus per-pixel camera noise. **Absolute brightness therefore says
nothing.** A pixel at 0.62 may be bright coating or it may be something on top
of the coating; only its value *relative to its immediate neighbourhood*
distinguishes the two. This is the same property the wafer images in lesson 10
had, and for the same physical reason.

A **blemish** is a speck of contamination baked in under the coating: a small
bright Gaussian blob about two pixels across, drawn by one function,
``_draw_blemish``, with one fixed contrast and one fixed spread. It looks
exactly the same in both batches, because it is the same code with the same
constants. A blemish centre never lands within ``TRUE_MARK_MARGIN`` pixels of
the patch border, so the whole blob is always visible and no model can succeed
by watching the edge of the image.


Station 1 — the coating grade
-----------------------------

The first station asks the obvious question: **did this window come out of the
coater clean?** Half the windows carry one blemish and half carry none, and the
station's grade is ``1`` for a blemished window and ``0`` for a clean one.

Where the blemish sits is not part of the question and is not recorded. It is
drawn uniformly over every legal position on the glass, and two windows with
the same blemish in different places get the same grade, because they are the
same event: the coater spat.


Station 2 — the aperture grade
------------------------------

Every window that station 1 rejects goes to a second bench, and this is where
the money is. A blemished window is not necessarily scrap. The sensor's beam
only passes through a **clear aperture** in the middle of the glass — the rest
of the disc is under the mounting flange and never sees light. A blemish
inside the clear aperture scatters the beam and the window is scrapped; a
blemish out in the flange region is cosmetic, and the window is shipped at
full price.

So the batch from station 2 contains **only blemished windows** — every single
image has exactly one blemish, drawn by the same function, with the same
contrast and the same spread, over the same legal positions. Asking whether a
blemish is present is not a question here; the answer is always yes. The grade
is ``1`` for scrap and ``0`` for ship.

The clear aperture is not drawn on the glass, not marked, and not visible in
the photograph. It is a property of the *mount*, not of the window, and it is
the same aperture on every unit in the batch.


What to do with all this
------------------------

Before you fit anything, read the two station descriptions again and ask the
question exercise 10 is built around.

Lesson 10 section 4.2 lists three things pooling does, and the third is that it
**discards spatial precision, which is a cost, not a benefit, whenever *where*
is part of the answer**. Section 6 then argues that a convolutional layer is a
*restricted* dense layer — strictly less expressive — and that the restriction
is the whole value, because "the functions it gives up are precisely the ones
that treat row 5 differently from row 15, and on this problem nobody wanted
any of those".

On which of these two stations does somebody want one of those functions?

The answer is in the two paragraphs above, not in the name of any architecture,
and a plot of the average blemished image from each batch will show it to you
in about a minute.

The ``TRUE_*`` constants below record exactly how each batch was built,
including the 3 % of grades each station records as the opposite of the truth.
That puts a **ceiling of 0.97 on both stations**. Note that this is *not* the
0.98 of the lesson-10 handout: the wafer grader ran at 2 %, these two benches
run at 3 %, and every accuracy in this exercise is to be read against 0.97.

**Do not read past the loader functions until Part 4.**
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared between the two stations, and held fixed on purpose
# ---------------------------------------------------------------------------

TRUE_IMAGE_SIZE = 24                 #: patches are 24 x 24, single channel
TRUE_N_IMAGES = 3000                 #: windows in each batch
TRUE_BLEMISH_RATE = 0.5              #: station 1 only: half the windows are blemished

#: Fraction of grades each station records as the opposite of the truth.
#: Identical on both stations, so it cannot explain any difference between
#: them. NOTE this is 3 %, not the wafer grader's 2 % from the handout.
TRUE_GRADER_ERROR_RATE = 0.03

#: The best accuracy any classifier can reach against the *recorded* grades,
#: even one that knows the true rule exactly: 1 - TRUE_GRADER_ERROR_RATE.
TRUE_BAYES_ACCURACY = 1.0 - TRUE_GRADER_ERROR_RATE

# --- the coated surface the blemish sits on ---------------------------------
#: Coating thickness varies smoothly across the glass, which is why the
#: background is not flat: a detector that keys on absolute brightness fails.
TRUE_BACKGROUND_LEVEL = 0.50
TRUE_BACKGROUND_SWING = 0.14         #: amplitude of the smooth swell
TRUE_SENSOR_NOISE = 0.045            #: per-pixel camera noise

# --- blemish appearance, identical at both stations -------------------------
TRUE_BLEMISH_SIGMA = 0.95            #: pixels, the blob's spread
TRUE_BLEMISH_CONTRAST = 0.46         #: blemishes are brighter than the coating

#: A blemish centre never lands closer than this to the patch border, so the
#: whole blob is visible and no model can grade a window by watching the edge.
TRUE_MARK_MARGIN = 5

#: Row bands used by the translation experiment of handout section 6, as
#: (low, high) centre rows. Station 1 only.
TRUE_TOP_BAND = (5, 11)
TRUE_BOTTOM_BAND = (13, 19)


def _background(rng, n, size):
    """Smooth coating-thickness swell plus camera noise.

    Two low-frequency sinusoids with random orientation and phase, so that no
    two windows have the same background and none of them is flat.
    """
    grid = np.arange(size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    out = np.empty((n, size, size))
    for i in range(n):
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


def _draw_blemish(image, row, col):
    """One small bright Gaussian blob centred on (row, col).

    The only blemish painter in this module. Both stations call it, with the
    same contrast and the same spread, so a blemish photographed at station 1
    and a blemish photographed at station 2 are indistinguishable as pictures.
    """
    size = image.shape[0]
    grid = np.arange(size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    squared = (yy - row) ** 2 + (xx - col) ** 2
    image += TRUE_BLEMISH_CONTRAST * np.exp(
        -squared / (2 * TRUE_BLEMISH_SIGMA ** 2))
    return image


def _sample_centre(rng, n, size, band=None):
    """Blemish centres: uniform over every legal position, or inside a band."""
    low, high = TRUE_MARK_MARGIN, size - TRUE_MARK_MARGIN
    cols = rng.integers(low, high, n)
    if band is None:
        rows = rng.integers(low, high, n)
    else:
        rows = rng.integers(band[0], band[1], n)
    return rows, cols


# ---------------------------------------------------------------------------
# The two loaders. Everything you need for Parts 1 to 3 is above this line.
# ---------------------------------------------------------------------------

def load_coating_grade(n_images=TRUE_N_IMAGES, seed=10101,
                       grader_error_rate=TRUE_GRADER_ERROR_RATE, band=None):
    """Station 1: is this window blemished at all?

    ``n_images`` windows, of which about half carry one blemish and the rest
    carry none. A blemished window is graded ``1``.

    ``band`` restricts where a blemish may appear: ``None`` for anywhere on
    the legal area, ``TRUE_TOP_BAND`` or ``TRUE_BOTTOM_BAND`` for one half of
    it. Training on one band and testing on the other is the translation
    experiment of handout section 6, and it is available here because at this
    station the position of the blemish is not part of the question.

    Returns ``(X, y, truth)``:

        X      float array (n_images, 24, 24), values in [0, 1]
        y      the *recorded* grade, 0 or 1 — the only labels a model may see
        truth  the noise-free grade, for Part 4 and for nothing else

    ``truth`` exists so that Part 4 can separate the bench's 3 % from a model's
    own mistakes — a separation no real batch record ever allows.
    """
    rng = np.random.default_rng(seed)
    size = TRUE_IMAGE_SIZE
    X = _background(rng, n_images, size)

    truth = (rng.random(n_images) < TRUE_BLEMISH_RATE).astype(int)
    rows, cols = _sample_centre(rng, n_images, size, band)
    for i in np.where(truth == 1)[0]:
        _draw_blemish(X[i], rows[i], cols[i])

    flip = rng.random(n_images) < grader_error_rate
    recorded = np.where(flip, 1 - truth, truth)
    return np.clip(X, 0.0, 1.0), recorded, truth


def load_aperture_grade(n_images=TRUE_N_IMAGES, seed=10102,
                        grader_error_rate=TRUE_GRADER_ERROR_RATE):
    """Station 2: this window is blemished — can it still be shipped?

    ``n_images`` windows, **every one of them carrying exactly one blemish**,
    drawn by the same painter as station 1 with the same contrast and the same
    spread, over the same legal positions. A window whose blemish falls inside
    the clear aperture is scrapped and graded ``1``; one whose blemish falls
    out in the flange region ships and is graded ``0``. The aperture itself is
    invisible in the photograph.

    The exact geometry is in the constants at the foot of this module, which
    are Part 4's business and not Part 1's.

    Returns ``(X, y, truth)`` with the same meanings as ``load_coating_grade``.
    """
    rng = np.random.default_rng(seed)
    size = TRUE_IMAGE_SIZE
    X = _background(rng, n_images, size)

    rows, cols = _sample_centre(rng, n_images, size, None)
    for i in range(n_images):
        _draw_blemish(X[i], rows[i], cols[i])

    truth = _inside_aperture(rows, cols).astype(int)

    flip = rng.random(n_images) < grader_error_rate
    recorded = np.where(flip, 1 - truth, truth)
    return np.clip(X, 0.0, 1.0), recorded, truth


def pixel_permutation(seed=10103, n_pixels=TRUE_IMAGE_SIZE ** 2):
    """One fixed shuffling of the 576 pixel positions, for the permutation test.

    Apply the *same* permutation to every image in a batch, train, and test.
    Handout section 9 does exactly this on the wafer images and separates two
    assumptions a convolution makes — weight sharing and locality — by noticing
    that only one of them survives. Use it the same way here:

        perm = pixel_permutation()
        X_shuffled = X.reshape(len(X), -1)[:, perm].reshape(X.shape)
    """
    return np.random.default_rng(seed).permutation(n_pixels)


# ===========================================================================
#  PART 4 STARTS HERE. Everything below gives the aperture rule away.
# ===========================================================================
#
# The clear aperture is a disc, concentric with the 24 x 24 patch. On a patch
# with an even number of pixels the centre falls between pixels, at 11.5 in
# both axes; a blemish centred on integer pixel (row, col) is inside the
# aperture when its distance from that point is below the radius.
#
# The radius was chosen to split the legal blemish positions as close to in
# half as the integer grid allows, so that this station's base rate matches
# station 1's and no comparison between the two can be explained by one of
# them having an easier majority-class baseline.

#: Centre of the clear aperture, in pixel coordinates, both axes.
TRUE_APERTURE_CENTRE = (TRUE_IMAGE_SIZE - 1) / 2.0     # 11.5

#: Radius of the clear aperture, in pixels. Legal blemish centres are the 196
#: integer positions in [5, 19) x [5, 19); 96 of them lie within 5.6 pixels of
#: (11.5, 11.5) and 100 do not, a design reject rate of 0.4898. That is as
#: close to an even split as the integer grid allows: the next ring of
#: positions out sits at distance 5.7009, and admitting it would take the
#: count to 112 and the rate to 0.5714.
TRUE_APERTURE_RADIUS_PX = 5.6


def _inside_aperture(rows, cols):
    """True where a blemish centred on (row, col) falls in the clear aperture.

        (row - 11.5)^2 + (col - 11.5)^2 < TRUE_APERTURE_RADIUS_PX^2

    Note what this rule is and is not. It is a condition on **where the
    blemish is**, and on nothing else: not on how bright it is, not on how big
    it is, not on what the coating underneath it is doing. Two windows with
    pixel-for-pixel identical blemishes, one at (11, 11) and one at (6, 6),
    are graded opposite ways.
    """
    centre = TRUE_APERTURE_CENTRE
    squared = (np.asarray(rows) - centre) ** 2 + (np.asarray(cols) - centre) ** 2
    return squared < TRUE_APERTURE_RADIUS_PX ** 2


def aperture_mask(size=TRUE_IMAGE_SIZE):
    """The clear aperture as a boolean 24 x 24 image, for Part 4 question 9.

    ``True`` where a blemish centred on that pixel would scrap the window.
    This is the whole rule, as a picture. Part 4 asks you to turn it into a
    classifier that has no parameters and needs no training at all.
    """
    grid = np.arange(size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    return _inside_aperture(yy, xx)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, loader in (("coating grade (station 1)", load_coating_grade),
                         ("aperture grade (station 2)", load_aperture_grade)):
        X, y, truth = loader()
        agree = (y == truth).mean()
        print(f"{name}: {len(X)} images of "
              f"{X.shape[1]}x{X.shape[2]}, range "
              f"[{X.min():.3f}, {X.max():.3f}]")
        print(f"  recorded reject rate {y.mean():.4f}, "
              f"true reject rate {truth.mean():.4f}")
        print(f"  majority-class baseline {max(y.mean(), 1 - y.mean()):.4f}")
        print(f"  recorded grade agrees with truth on {agree:.4f} of windows")
        print()

    mask = aperture_mask()
    legal = np.zeros_like(mask)
    legal[TRUE_MARK_MARGIN:TRUE_IMAGE_SIZE - TRUE_MARK_MARGIN,
          TRUE_MARK_MARGIN:TRUE_IMAGE_SIZE - TRUE_MARK_MARGIN] = True
    print(f"legal blemish positions: {legal.sum()}")
    print(f"  of which inside the aperture: {(mask & legal).sum()}")
    print(f"  design reject rate at station 2: "
          f"{(mask & legal).sum() / legal.sum():.4f}")
