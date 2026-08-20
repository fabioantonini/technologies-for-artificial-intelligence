"""Two inspection stations on a polymer compounding line, for exercise 6.

A compounding plant buys polymer pellets by the sack, melts them in a twin-screw
extruder and sells the compounded product against a melt-flow specification.
Two things can go wrong, they are measured by two completely different
instruments, and — this is the whole point of the exercise — **they are wrong in
two completely different shapes**.

``load_spectra``    goods-inwards screening. 80 near-infrared channels per sack,
                    only a couple of hundred sacks ever confirmed by the lab.
                    Wide, short, and most of the columns contain nothing.

``load_extrusion``  the extruder's own energy monitor. Two columns, nine hundred
                    runs, and neither column means anything on its own.

Lesson 6 compared three families on one dataset and the ranking came out
0.613 / 0.933 / 0.944 / 0.947. Here the ranking is different at each station,
and neither ordering can be guessed from the names of the algorithms. Look at
the data first. That is the exercise.


Station A — the near-infrared spectrometer
------------------------------------------

Every incoming sack is scanned by an indium-gallium-arsenide (InGaAs) diode
array: 80 photodiodes, one per wavelength, from 1100 nm to 1692.5 nm in steps of
7.5 nm (``WAVELENGTHS_NM``). The reading is absorbance, and the instrument
applies its own baseline and scatter correction in firmware, so what reaches the
file is already detrended.

The failure mode is a **grade mix-up**: a sack of the recycled grade finds its
way onto a pallet of virgin material. It is not a gradual drift — a sack is
either the wrong grade or it is not, and every wrong sack is the *same* wrong
material. The recycled grade absorbs a little more light across one band of
wavelengths and is indistinguishable everywhere else.

Three consequences, and they are what make this station worth studying.

1. **Within a class, the channels vary only because of detector noise**, and
   each photodiode has its own noise. So given the class, the 80 readings are
   independent. The assumption Naive Bayes makes is not approximately true here,
   it is exactly true — which is rare, and is why this station exists. Do not
   take that on trust: measure it.
2. **No single channel is worth much.** The difference the contaminant makes on
   the best channel is under one detector noise unit. The evidence has to be
   accumulated across many channels or it is not there at all.
3. **The lab reference is expensive.** A wet-chemistry confirmation costs a day
   and a technician, so only a couple of hundred sacks a year get a label, and
   the labels are the lab's answer rather than the truth — the reference method
   has an error rate of its own. That sets a ceiling on any accuracy measured
   here.

The two obvious peaks in the average spectrum are the polymer's own absorbance.
They are in both classes and they carry nothing.


Station B — the extruder energy monitor
----------------------------------------

Each production run records throughput in kg/h and heater power in kW. What the
process actually cares about is neither: it is the **specific energy input**,
power divided by throughput, in kWh/kg. Too little and the pellets do not fully
melt; too much and the polymer degrades in the barrel. Both give product outside
the melt-flow specification, so the good region is a *band* of ratios and the
bad region is everything on either side of it.

In the plane the two columns are measured in, that band is a wedge, and no
straight line separates a wedge from its complement. Neither column on its own
says much: the same throughput appears in good and bad runs alike, and the same
heater power does too.


Both loaders are offline, seeded, and give the same numbers anywhere.

The ``TRUE_*`` constants below record how the labels were generated: which
channels the contaminant actually touches, which one is dead, and where the
energy window sits. **Do not read them until Part 5.** Reading them afterwards,
to mark yourself, is the reason they are written down.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 23

# ---------------------------------------------------------------------------
# Station A — the spectrometer
# ---------------------------------------------------------------------------

N_CHANNELS = 80
CHANNEL_NAMES = [f"nir_{j:02d}" for j in range(N_CHANNELS)]

#: Wavelength of each photodiode, in nanometres. Flavour for the plots; the
#: classifiers never see it.
WAVELENGTHS_NM = 1100.0 + 7.5 * np.arange(N_CHANNELS)

#: Detector noise on a nominal photodiode, in absorbance units. Each diode's
#: own gain multiplies it — see ``CHANNEL_GAIN``.
BASE_NOISE = 0.010

# --- published truth: do not read before Part 5 ----------------------------

#: The wavelength band the recycled grade absorbs in, as channel indices.
TRUE_BAND = tuple(range(18, 42))

#: Centre and width of the absorption feature within that band, in channels.
TRUE_BAND_CENTRE = 29.5
TRUE_BAND_WIDTH = 11.0

#: One photodiode inside the band sits on a manufacturing defect. The
#: instrument masks it and interpolates, so it carries detector noise and no
#: contaminant signal whatsoever. This is the decoy: it is in the middle of the
#: real band and it is worth exactly nothing.
TRUE_DEAD_CHANNEL = 26

#: Total separation between the two classes, as a Mahalanobis distance summed
#: over all channels. Everything a perfect classifier could know is in here:
#: the best achievable accuracy against the true state is Phi(d / 2).
TRUE_SEPARATION = 3.20

#: Fraction of incoming sacks that are the wrong grade.
TRUE_CONTAMINATION_RATE = 0.42

#: Error rate of the lab reference method. The ``contaminated`` column is the
#: lab's answer, not the truth, so this caps the accuracy of anything measured
#: against it.
TRUE_LAB_ERROR_RATE = 0.03

# ---------------------------------------------------------------------------


def _channel_gain() -> np.ndarray:
    """Per-photodiode gain. Fixed by the detector, not by the sample."""
    rng = np.random.default_rng(RANDOM_STATE + 101)
    return np.clip(np.exp(rng.normal(0.0, 0.32, N_CHANNELS)), 0.55, 2.0)


#: Noise standard deviation of each channel, in absorbance units. It varies by
#: a factor of about three across the array, which is why Euclidean distance on
#: the raw columns is not the same computation as on standardised ones.
CHANNEL_GAIN = _channel_gain()
CHANNEL_NOISE = BASE_NOISE * CHANNEL_GAIN


def _baseline_spectrum() -> np.ndarray:
    """The virgin polymer's own absorbance. Identical in both classes."""
    j = np.arange(N_CHANNELS)
    return (0.42
            + 0.34 * np.exp(-((j - 54.0) / 11.0) ** 2)
            + 0.21 * np.exp(-((j - 11.0) / 7.0) ** 2))


BASELINE = _baseline_spectrum()


def _effect_size() -> np.ndarray:
    """How far the contaminant moves each channel, in that channel's noise units.

    A smooth absorption peak over ``TRUE_BAND``, scaled so that the total
    separation is ``TRUE_SEPARATION``, with the dead channel zeroed.
    """
    profile = np.zeros(N_CHANNELS)
    band = np.array(TRUE_BAND)
    profile[band] = np.exp(-((band - TRUE_BAND_CENTRE) / TRUE_BAND_WIDTH) ** 2)
    profile[TRUE_DEAD_CHANNEL] = 0.0
    return profile * (TRUE_SEPARATION / np.sqrt((profile ** 2).sum()))


#: Per-channel effect size. Zero on 57 of the 80 channels.
TRUE_EFFECT_SIZE = _effect_size()


def load_spectra(n_samples: int = 200, random_state: int = RANDOM_STATE):
    """Near-infrared scans of incoming sacks, and the lab's verdict on each.

    Returns ``(X, y)``: a DataFrame of 80 absorbance columns and a Series
    ``contaminated`` of zeros and ones. About 42% of sacks are the wrong grade.

    ``n_samples`` is how many sacks the lab managed to confirm. The default,
    200, is one year of the plant's actual budget; larger values are what the
    plant would have after several years, and the exercise asks you to measure
    what changes.
    """
    rng = np.random.default_rng(random_state)

    contaminated = rng.random(n_samples) < TRUE_CONTAMINATION_RATE
    centres = (BASELINE[None, :]
               + np.outer(contaminated, TRUE_EFFECT_SIZE * CHANNEL_NOISE))
    readings = centres + rng.normal(0.0, 1.0,
                                    (n_samples, N_CHANNELS)) * CHANNEL_NOISE

    mistaken = rng.random(n_samples) < TRUE_LAB_ERROR_RATE
    reported = np.where(mistaken, ~contaminated, contaminated)

    X = pd.DataFrame(readings.round(5), columns=CHANNEL_NAMES)
    return X, pd.Series(reported.astype(int), name="contaminated")


# ---------------------------------------------------------------------------
# Station B — the extruder energy monitor
# ---------------------------------------------------------------------------

# --- published truth: do not read before Part 5 ----------------------------

#: The specific energy window, in kWh/kg. Below it the pellets do not melt;
#: above it the polymer degrades. Both are off specification.
TRUE_ENERGY_WINDOW = (0.16, 0.30)

#: Range of specific energy the line actually runs at, in kWh/kg.
TRUE_ENERGY_RANGE = (0.10, 0.37)

#: Range of throughput, in kg/h. Chosen independently of the energy, so
#: throughput on its own says nothing about the outcome at all.
TRUE_THROUGHPUT_RANGE = (45.0, 145.0)

#: Relative error of the two meters, and the rate at which the melt-flow test
#: itself disagrees with the process. Together these set the ceiling.
TRUE_METER_ERROR = 0.010
TRUE_SPEC_TEST_NOISE = 0.03

# ---------------------------------------------------------------------------


def load_extrusion(n_samples: int = 900, random_state: int = RANDOM_STATE):
    """One row per production run: what went in, and whether it met spec.

    Returns ``(X, y)`` with ``X`` a DataFrame of ``throughput_kg_h`` and
    ``heater_power_kw``, and ``y`` a Series ``off_spec``. Roughly 48% of runs
    are off specification.
    """
    rng = np.random.default_rng(random_state + 7)

    throughput = rng.uniform(*TRUE_THROUGHPUT_RANGE, n_samples)
    energy = rng.uniform(*TRUE_ENERGY_RANGE, n_samples)
    power = energy * throughput

    low, high = TRUE_ENERGY_WINDOW
    off_spec = ((energy < low) | (energy > high)).astype(int)
    flipped = rng.random(n_samples) < TRUE_SPEC_TEST_NOISE
    off_spec = np.where(flipped, 1 - off_spec, off_spec)

    # What the meters report is not quite what the process did.
    throughput = throughput * (1.0 + rng.normal(0.0, TRUE_METER_ERROR, n_samples))
    power = power * (1.0 + rng.normal(0.0, TRUE_METER_ERROR, n_samples))

    X = pd.DataFrame({"throughput_kg_h": throughput.round(2),
                      "heater_power_kw": power.round(3)})
    return X, pd.Series(off_spec, name="off_spec")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Xa, ya = load_spectra()
    print(f"station A: {Xa.shape[0]} sacks, {Xa.shape[1]} channels, "
          f"{int(ya.sum())} labelled contaminated ({ya.mean():.1%})")
    print(f"  wavelengths {WAVELENGTHS_NM[0]:.0f}-{WAVELENGTHS_NM[-1]:.0f} nm")
    print(f"  absorbance {Xa.to_numpy().min():.3f} to {Xa.to_numpy().max():.3f}")
    separations = (Xa.groupby(ya).mean().diff().iloc[-1].abs()
                   / Xa.std(ddof=0))
    print(f"  best single channel separates the classes by "
          f"{separations.max():.2f} standard deviations "
          f"({separations.idxmax()})")

    Xb, yb = load_extrusion()
    print(f"\nstation B: {Xb.shape[0]} runs, {Xb.shape[1]} columns, "
          f"{int(yb.sum())} off specification ({yb.mean():.1%})")
    print(f"  throughput {Xb['throughput_kg_h'].min():.0f}-"
          f"{Xb['throughput_kg_h'].max():.0f} kg/h, "
          f"power {Xb['heater_power_kw'].min():.1f}-"
          f"{Xb['heater_power_kw'].max():.1f} kW")
    print("  correlation of each column with the label: "
          + ", ".join(f"{name} {value:+.3f}"
                      for name, value in Xb.corrwith(yb).items()))
