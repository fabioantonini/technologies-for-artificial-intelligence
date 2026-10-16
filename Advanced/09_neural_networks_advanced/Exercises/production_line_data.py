"""Two production lines from the same factory, for exercise 9.

Meridian Instruments — the fictional maker of optical distance sensors the
lesson-9 notebooks use — runs two final-test stations, one per product line.
Both stations do the same job: measure **eight** numbers on a finished unit
and record a single pass/fail verdict. This module gives you one batch from
each station, deliberately matched so that almost nothing differs between
them:

===========================  =============  =============
                             lens assembly  burn-in screen
===========================  =============  =============
units in the batch                   3,000          3,000
measured columns                         8              8
pass rate                          ~0.50          ~0.50
verdicts recorded wrongly             3 %            3 %
===========================  =============  =============

Same size, same number of columns, same base rate, same label noise. The one
thing that is *not* the same is **the shape of the rule that decides the
verdict**, and that is the whole exercise.

``load_lens_assembly``   an optical bench line: a lens cell is pressed into a
                         machined barrel.
``load_burn_in_screen``  a laser-diode line: finished modules are screened for
                         early-life failure.

Neither loader touches the network. Both generate their batch locally with
NumPy and return the same numbers on any machine.


The lens assembly line
----------------------

A ground lens cell is pressed into a machined aluminium barrel. Both parts
come off their own grinder with their own tolerance, and both are measured
before assembly, as deviations from nominal in micrometres (µm):

* ``barrel_bore_um`` — how much wider or narrower the barrel's bore is than
  the drawing says.
* ``cell_od_um`` — how much wider or narrower the lens cell's outer diameter
  is than the drawing says.

The press fit is good when the **clearance between the two**, that is the bore
minus the outer diameter, lands inside a narrow band around nominal. Too much
clearance and the cell rattles and the optical axis wanders; too little and
pressing it home stresses the glass and the coating crazes. So the assembly
passes when the clearance is small in magnitude — in *either* direction.

Notice what this does to each part on its own. A barrel bored 8 µm wide is
perfectly fine if it meets a cell ground 8 µm oversize, and scrap if it meets a
nominal one. **Neither measurement means anything by itself**; only the pair
does. The station also records six further numbers — ambient temperature,
press force, cell mass, coating thickness, barrel length and bench humidity —
because the line's quality system requires it. None of the six has any bearing
on whether the fit is good, and they are generated here independently of
everything else.


The burn-in screen
------------------

Finished laser-diode modules are run hot for forty-eight hours and the ones
likely to fail in the field are pulled. Eight electro-optical indicators are
measured on each module before the run: threshold current, slope efficiency,
series resistance, dark current, wavelength offset, facet temperature rise,
wire-bond pull strength and emission ripple.

Reliability engineering says none of these individually condemns a module.
Each one is a *small independent contribution to a hazard score*: a slightly
high threshold current adds a little risk, a slightly low bond pull strength
adds a little more, and a module fails the screen when the accumulated total
crosses the line's limit. That is exactly how a composite screening
specification is written, and all eight indicators contribute to it — some
more heavily than others, none of them decisively.

So on this line every column carries a little of the answer on its own, and
what the station is really doing is adding up eight small votes.


What to do with all this
------------------------

Before you fit anything, read the two descriptions above again and ask the
question exercise 9 is built around: on which line does the verdict depend on
the measurements **in combination**, so that no weighted sum of the columns
could express it — and on which line is the verdict a weighted sum by
construction? The handout's section 12 has the table; the answer is in these
two paragraphs, not in the name of any model.

The ``TRUE_*`` constants below record exactly how each batch was built,
including the 3 % of verdicts each station records as the opposite of the
truth. That last one puts a **ceiling of 0.97 on both lines**, so read every
accuracy in this exercise against 0.97 rather than against 1.000 — the same
convention lesson 9's own acceptance data uses.

**Do not read past the loader functions until Part 4.**
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared between the two lines, and held fixed on purpose
# ---------------------------------------------------------------------------

#: Units in each batch. The same on both lines, so no comparison below can be
#: explained by one batch being bigger than the other.
TRUE_N_UNITS = 3000

#: Fraction of verdicts each station records as the opposite of the truth.
#: Identical on both lines, again so that it cannot explain any difference.
TRUE_RIG_ERROR_RATE = 0.03

#: The best accuracy any classifier can reach against the *recorded* verdicts,
#: even one that knows the true rule exactly: 1 - TRUE_RIG_ERROR_RATE.
TRUE_BAYES_ACCURACY = 1.0 - TRUE_RIG_ERROR_RATE


# ---------------------------------------------------------------------------
# 1. Lens assembly — the verdict depends on two columns jointly
# ---------------------------------------------------------------------------
#
# Both mating dimensions are ground to nominal with the same production
# spread, so the clearance between them is centred on zero. The assembly
# passes when |bore - outer diameter| is inside the tolerance band, which is a
# *strip* in the plane of the two measurements: two parallel boundaries, not
# one. That is the point of this line.

#: Production spread of each mating dimension, in micrometres.
TRUE_BORE_SIGMA_UM = 6.0
TRUE_OD_SIGMA_UM = 6.0

#: Half-width of the clearance tolerance band, in micrometres. Chosen so that
#: almost exactly half of all assemblies pass: the clearance has standard
#: deviation sqrt(6^2 + 6^2) = 8.4853 µm, and 5.7 / 8.4853 = 0.6717, which
#: is close to the 0.6745 that would split a normal population exactly in two.
TRUE_CLEARANCE_TOL_UM = 5.7

#: The two columns that decide the verdict. Neither is informative alone.
TRUE_FIT_FEATURES = ("barrel_bore_um", "cell_od_um")

#: The six columns the quality system records and the rule ignores entirely.
#: Each is drawn independently of the label and of every other column.
TRUE_LENS_NUISANCE = {
    #                          mean,   spread
    "ambient_temp_c":          (21.0,  1.2),
    "press_force_n":           (140.0, 12.0),
    "cell_mass_g":             (4.2,   0.15),
    "coating_thickness_nm":    (215.0, 9.0),
    "barrel_length_um":        (0.0,   15.0),
    "bench_humidity_pct":      (44.0,  5.0),
}

LENS_FEATURES = list(TRUE_FIT_FEATURES) + list(TRUE_LENS_NUISANCE)


def load_lens_assembly(n_units=TRUE_N_UNITS, rig_error_rate=TRUE_RIG_ERROR_RATE,
                       random_state=9101):
    """One batch off the lens assembly line: 3,000 units, eight measurements.

    The two mating dimensions are drawn independently from centred normal
    distributions with the production spreads above. The assembly is *truly*
    within tolerance when

        abs(barrel_bore_um - cell_od_um) < TRUE_CLEARANCE_TOL_UM

    and the station records that verdict correctly with probability
    1 - rig_error_rate, flipping it otherwise.

    Returns a DataFrame with ten columns:

        the eight measurements  the features a model may use
        assembly_passes         the *recorded* verdict (noisy) — 0/1
        truly_within_tolerance  the noise-free verdict, for checking only

    A model is only ever shown the eight measurements. The last column exists
    so that Part 4 can separate the station's 3 % from the model's own
    mistakes — a separation no real batch record ever allows.
    """
    rng = np.random.default_rng(random_state)

    bore = rng.normal(0.0, TRUE_BORE_SIGMA_UM, n_units)
    od = rng.normal(0.0, TRUE_OD_SIGMA_UM, n_units)

    truth = (np.abs(bore - od) < TRUE_CLEARANCE_TOL_UM).astype(int)

    flip = rng.random(n_units) < rig_error_rate
    recorded = np.where(flip, 1 - truth, truth)

    columns = {"barrel_bore_um": bore, "cell_od_um": od}
    for name, (mean, spread) in TRUE_LENS_NUISANCE.items():
        columns[name] = rng.normal(mean, spread, n_units)

    df = pd.DataFrame(columns)
    df[LENS_FEATURES] = df[LENS_FEATURES].round(3)
    df["assembly_passes"] = recorded
    df["truly_within_tolerance"] = truth
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Burn-in screen — the verdict is a weighted sum of all eight columns
# ---------------------------------------------------------------------------
#
# Each indicator is measured in its own units and on its own scale. The
# hazard score adds them up after standardising each one, which is what a
# composite screening specification does; the module survives when the total
# stays below the line's limit.

#: Each indicator's mean and production spread, in its own physical units.
TRUE_INDICATOR_SCALES = {
    #                              mean,   spread
    "threshold_current_ma":        (38.0,  4.0),
    "slope_efficiency_w_per_a":    (0.72,  0.06),
    "series_resistance_ohm":       (2.40,  0.30),
    "dark_current_na":             (11.0,  2.5),
    "wavelength_offset_nm":        (0.0,   1.5),
    "facet_temp_rise_c":           (6.5,   1.1),
    "bond_pull_strength_gf":       (48.0,  5.0),
    "emission_ripple_pct":         (2.1,   0.5),
}

#: Weight each *standardised* indicator carries in the hazard score. Positive
#: means "more of this is worse". Every one of the eight is non-zero: this is
#: a vote, and no single indicator wins it. The largest weight is 1.1 and the
#: smallest 0.3, a factor of under four between them.
TRUE_HAZARD_WEIGHTS = {
    "threshold_current_ma":      1.0,
    "slope_efficiency_w_per_a": -0.8,
    "series_resistance_ohm":     0.9,
    "dark_current_na":           1.1,
    "wavelength_offset_nm":      0.3,
    "facet_temp_rise_c":         0.7,
    "bond_pull_strength_gf":    -0.6,
    "emission_ripple_pct":       0.5,
}

#: The line's screening limit, on the hazard score. A module survives when its
#: score is below this. Zero, because every standardised indicator is centred
#: on zero and this splits the population in half — matching the lens line's
#: pass rate on purpose.
TRUE_HAZARD_LIMIT = 0.0

BURN_IN_FEATURES = list(TRUE_INDICATOR_SCALES)


def load_burn_in_screen(n_units=TRUE_N_UNITS, rig_error_rate=TRUE_RIG_ERROR_RATE,
                        random_state=9102):
    """One batch off the laser-diode line: 3,000 modules, eight indicators.

    Every indicator is drawn independently from its own normal distribution.
    The hazard score is the weighted sum of the eight *standardised*
    indicators,

        score = sum_j TRUE_HAZARD_WEIGHTS[j] * (x_j - mean_j) / spread_j

    and the module *truly* survives the screen when ``score <
    TRUE_HAZARD_LIMIT``. The station records that verdict correctly with
    probability 1 - rig_error_rate, flipping it otherwise.

    Returns a DataFrame with ten columns:

        the eight indicators    the features a model may use
        survives_burn_in        the *recorded* verdict (noisy) — 0/1
        truly_below_limit       the noise-free verdict, for checking only

    Note what the rule is and is not. It is a threshold on a weighted sum of
    the columns as measured — no products, no ratios, no absolute values. It
    is *deterministic*: the only randomness in the recorded verdict is the
    station's 3 %, exactly as on the lens line.
    """
    rng = np.random.default_rng(random_state)

    columns = {}
    score = np.zeros(n_units)
    for name, (mean, spread) in TRUE_INDICATOR_SCALES.items():
        z = rng.normal(0.0, 1.0, n_units)
        columns[name] = mean + spread * z
        score += TRUE_HAZARD_WEIGHTS[name] * z

    truth = (score < TRUE_HAZARD_LIMIT).astype(int)

    flip = rng.random(n_units) < rig_error_rate
    recorded = np.where(flip, 1 - truth, truth)

    df = pd.DataFrame(columns)
    df[BURN_IN_FEATURES] = df[BURN_IN_FEATURES].round(3)
    df["survives_burn_in"] = recorded
    df["truly_below_limit"] = truth
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lens = load_lens_assembly()
    burn = load_burn_in_screen()

    for name, df, label, truth_col, features in (
        ("lens assembly", lens, "assembly_passes",
         "truly_within_tolerance", LENS_FEATURES),
        ("burn-in screen", burn, "survives_burn_in",
         "truly_below_limit", BURN_IN_FEATURES),
    ):
        agree = (df[label] == df[truth_col]).mean()
        print(f"{name}: {len(df)} units, {len(features)} measured columns")
        print(f"  recorded pass rate {df[label].mean():.4f}, "
              f"true pass rate {df[truth_col].mean():.4f}")
        print(f"  recorded verdict agrees with truth on {agree:.4f} of units")
        print(df[features].describe().loc[["mean", "std", "min", "max"]].round(3))
        print()
