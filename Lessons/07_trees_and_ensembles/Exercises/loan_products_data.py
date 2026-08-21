"""Two loan products from the same lender, for exercise 7.

Lesson 7 used one loan book — retail applicants, income and debt-to-income
ratio — where an underwriting checklist with a couple of "stressed"
combinations gave the ensembles a clear, consistent edge over any single
tree. This module gives you **two different products from the same lender**,
built the same way (a rule, applied, then 7% of labels flipped to stand in
for outcomes the two on-file features cannot explain — a job loss, a client
who pays late for reasons that have nothing to do with revenue). The two
products are underwritten by completely different logic, and that
difference is the entire exercise.

``load_personal_loan``    two features, a two-item checklist, drawable.
``load_equipment_loan``   three features, five disjoint "it depends"
                           pockets, none of them a single threshold.

Neither loader tells you which method wins. Read on and you can work it out
before fitting anything, which is exactly what Part 1 asks you to do.


Product 1 — the personal loan
------------------------------

Retail lending on a fixed schedule: a few thousand euros, repaid monthly.
Underwriting here is a **checklist that reads one factor at a time**:
decline anyone whose income cannot plausibly support any personal loan, and
decline anyone who has already committed most of their available revolving
credit, because a customer that stretched has no slack left for a new
monthly payment. Neither rule needs to know anything about the other.

The rule is the union of two independent half-planes: income below a floor,
**or** revolving-credit utilisation above a ceiling. A tree needs exactly
two splits to draw it — one per condition, in either order — and every row
that supports the split is a row anywhere on that side of the threshold, so
there is a lot of evidence behind each one.


Product 2 — the equipment loan
--------------------------------

Commercial lending against machinery: bigger tickets, and the borrower is a
small business rather than an individual. Three figures are on file at
application: the business's annual revenue, its debt-service coverage ratio
(DSCR — free cash flow available divided by the debt payments it must
cover; the industry's own summary number for "can this business afford
this loan"), and a sector-cyclicality score from 0 (steady demand,
groceries, utilities) to 1 (highly cyclical, construction, discretionary
retail).

There is no floor and no ceiling on any one of the three. A revenue level
that is comfortable in a steady sector is not automatically comfortable in
a volatile one; a DSCR that looks adequate on paper only actually is
adequate for some combinations of size and sector, not others. The
underwriting rule that follows from this is a set of five distinct
"combination" pockets, each occupying a modest box in the three-dimensional
space of (revenue, DSCR, sector volatility) — recognisable only as a
specific combination, not as any one number crossing any one line. **Every
pocket is a three-way condition: all three figures have to fall in a
specific range simultaneously**, and there is no shortcut through one
feature alone — read the ``TRUE_STRESS_POCKETS`` table below and you will
not find a single column where "high" or "low" alone says much (Part 3's
suggested measurements let you check that rather than take it on trust).

The two products are the same size (900 applicants each) so that sample
size is not what is doing the work below — it is held fixed on purpose.
What differs is **how many disjoint regions the rule needs, and how many
features have to agree at once to describe each one**. That, not the name
of any algorithm, is what should drive your prediction in Part 1.

The ``TRUE_*`` constants below record exactly how each rule was built.
**Do not read past the loader functions until Part 4.**

Both loaders are offline, seeded with NumPy's ``default_rng``, and return
the same numbers on any machine.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Product 1 — the personal loan
# ---------------------------------------------------------------------------

#: Range of applicant income simulated, in thousands of euros a year.
INCOME_RANGE = (15.0, 90.0)

#: Range of revolving-credit utilisation simulated, as a percentage of the
#: applicant's existing credit limit currently drawn down.
UTILISATION_RANGE = (0.0, 100.0)

# --- published truth: do not read before Part 4 ----------------------------

#: Below this income, in thousands of euros a year, decline regardless of
#: utilisation: there is no realistic repayment capacity for a new loan.
TRUE_INCOME_FLOOR = 25.0

#: Above this utilisation percentage, decline regardless of income: the
#: applicant has no free capacity left on their existing credit.
TRUE_UTILISATION_CEILING = 85.0

#: Fraction of labels flipped after the rule is applied — outcomes this
#: lender's two on-file features cannot explain (a job loss, an inheritance
#: that clears other debt). The same rate as lesson 7's own dataset.
TRUE_LABEL_NOISE_PERSONAL = 0.07

# ---------------------------------------------------------------------------


def load_personal_loan(n_samples: int = 900, random_state: int = RANDOM_STATE):
    """Personal-loan applicants: income, credit utilisation, and default.

    Returns ``(X, y)``: a DataFrame of ``income_k`` (thousands of euros a
    year) and ``utilisation_pct`` (0-100), and a Series ``default``. About
    33% of applicants default. The risky region is the union of two
    half-planes — an income floor and a utilisation ceiling — so a single
    threshold on either feature already recovers most of the rule, and a
    two-split tree recovers essentially all of it.
    """
    rng = np.random.default_rng(random_state)
    income = rng.uniform(*INCOME_RANGE, n_samples)
    utilisation = rng.uniform(*UTILISATION_RANGE, n_samples)

    default = (
        (income < TRUE_INCOME_FLOOR) | (utilisation > TRUE_UTILISATION_CEILING)
    ).astype(int)
    flip = rng.random(n_samples) < TRUE_LABEL_NOISE_PERSONAL
    default = np.where(flip, 1 - default, default)

    X = pd.DataFrame({
        "income_k": income.round(2),
        "utilisation_pct": utilisation.round(2),
    })
    return X, pd.Series(default, name="default")


# ---------------------------------------------------------------------------
# Product 2 — the equipment loan
# ---------------------------------------------------------------------------

#: Range of annual business revenue simulated, in thousands of euros.
REVENUE_RANGE = (30.0, 400.0)

#: Range of debt-service coverage ratio (DSCR) simulated. Above 1.0 means
#: free cash flow exceeds the debt payment; below 1.0 it does not cover it.
DSCR_RANGE = (0.70, 2.20)

#: Range of the sector-cyclicality score simulated: 0 is steady demand,
#: 1 is highly cyclical.
SECTOR_VOL_RANGE = (0.0, 1.0)

# --- published truth: do not read before Part 4 ----------------------------

#: Five disjoint "stress pockets" in (revenue_k, dscr, sector_vol) space.
#: Each is a box: default is 1 if and only if all three figures fall inside
#: the same row simultaneously. None of the three columns has a value that,
#: on its own, is "risky" — each range below also appears in at least one
#: *safe* combination elsewhere in the space. Together the five pockets
#: cover about 22% of the applicant population.
TRUE_STRESS_POCKETS = (
    # (revenue_k range,   DSCR range,    sector_vol range)  — plain-language read
    ((30.0, 165.0),  (1.65, 2.20), (0.65, 1.00)),  # small + strong coverage, but a volatile sector eats the buffer
    ((165.0, 300.0), (1.30, 1.85), (0.00, 0.35)),  # mid-size + comfortable coverage, in a steady sector that hides thin real margins
    ((100.0, 235.0), (0.70, 1.25), (0.35, 0.75)),  # small-to-mid + borderline coverage + middling volatility, all at once
    ((235.0, 370.0), (1.00, 1.55), (0.75, 1.00)),  # large + adequate-looking coverage, undone by a highly cyclical sector
    ((30.0, 165.0),  (0.95, 1.50), (0.00, 0.25)),  # small + adequate coverage, in a sector too steady to justify the price charged
)

#: Fraction of labels flipped after the rule is applied — the same
#: noise-floor logic and the same rate as the personal-loan product.
TRUE_LABEL_NOISE_EQUIPMENT = 0.07

# ---------------------------------------------------------------------------


def load_equipment_loan(n_samples: int = 900, random_state: int = RANDOM_STATE):
    """Equipment-loan applicants: revenue, coverage, sector, and default.

    Returns ``(X, y)``: a DataFrame of ``revenue_k`` (thousands of euros),
    ``dscr`` (debt-service coverage ratio) and ``sector_vol`` (0-1), and a
    Series ``default``. About 27% of applicants default. The risky region is
    the union of five disjoint boxes in the three-feature space, each
    requiring all three figures to fall in a specific range at once — there
    is no single threshold on any one column, or even any pair of columns,
    that comes close to recovering the rule alone.
    """
    rng = np.random.default_rng(random_state)
    revenue = rng.uniform(*REVENUE_RANGE, n_samples)
    dscr = rng.uniform(*DSCR_RANGE, n_samples)
    sector_vol = rng.uniform(*SECTOR_VOL_RANGE, n_samples)

    def _in_box(values, lo_hi):
        lo, hi = lo_hi
        return (values >= lo) & (values <= hi)

    default = np.zeros(n_samples, dtype=bool)
    for revenue_box, dscr_box, vol_box in TRUE_STRESS_POCKETS:
        default |= (
            _in_box(revenue, revenue_box)
            & _in_box(dscr, dscr_box)
            & _in_box(sector_vol, vol_box)
        )
    default = default.astype(int)

    flip = rng.random(n_samples) < TRUE_LABEL_NOISE_EQUIPMENT
    default = np.where(flip, 1 - default, default)

    X = pd.DataFrame({
        "revenue_k": revenue.round(2),
        "dscr": dscr.round(3),
        "sector_vol": sector_vol.round(3),
    })
    return X, pd.Series(default, name="default")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Xp, yp = load_personal_loan()
    print(f"personal loan: {len(Xp)} applicants, {int(yp.sum())} defaulted "
          f"({yp.mean():.1%}), majority baseline {max(yp.mean(), 1 - yp.mean()):.3f}")
    print(f"  income   {Xp['income_k'].min():.1f} to {Xp['income_k'].max():.1f} k EUR")
    print(f"  util.    {Xp['utilisation_pct'].min():.1f} to "
          f"{Xp['utilisation_pct'].max():.1f} %")

    Xe, ye = load_equipment_loan()
    print(f"\nequipment loan: {len(Xe)} applicants, {int(ye.sum())} defaulted "
          f"({ye.mean():.1%}), majority baseline {max(ye.mean(), 1 - ye.mean()):.3f}")
    print(f"  revenue    {Xe['revenue_k'].min():.1f} to {Xe['revenue_k'].max():.1f} k EUR")
    print(f"  dscr       {Xe['dscr'].min():.2f} to {Xe['dscr'].max():.2f}")
    print(f"  sector_vol {Xe['sector_vol'].min():.2f} to {Xe['sector_vol'].max():.2f}")
