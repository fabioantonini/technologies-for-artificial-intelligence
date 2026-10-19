---
title: "Lesson 3: Regression"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "16 October 2026"
---

# Before we start

- Exercise 2, which we discuss now
- The pipeline you built is the one we fit models into from now on

::: notes
Nothing is collected. Ask two or three people what they built. If the common
mistake appeared, name it: fitting the scaler or the imputer before splitting
the data. It is the exact thing lesson 2 warned about, and seeing it in
their own work is more instructive than any slide.

Then the connection forward: everything from today onwards goes inside that
ColumnTransformer. We are no longer preparing data, we are fitting models to
prepared data.
:::

# Today: the first real model

- The model and what it claims
- The exact solution, and why it sometimes fails
- Gradient descent
- Curves, and the price of flexibility
- Ridge and Lasso

::: notes
The agenda. Say that this is the first lesson where we fit a model, and that
regression is a deliberate first choice for three reasons: it has an exact
solution, so we can see what fitting means with no machinery in the way; it has
an iterative one too, which is the same algorithm that trains neural networks in
lesson 9; and its coefficients are readable in the units of the problem.

That third property is why linear models are still in production in medicine and
credit scoring, decades after more accurate methods existed. Worth saying - the
room usually assumes "simple" means "obsolete".
:::

# One dataset, all lesson

600 houses. Six measurements. A price.

**And we know the coefficients that generated it.**

In code: `train_test_split(X, y, test_size=0.25)` keeps 150 houses for testing.


::: notes
Synthetic, deliberately. Every estimate today can be checked against the truth,
which no real dataset allows.

Give the numbers: 2,400 euros per square metre, 9,000 per bedroom, 14,000 per
bathroom, minus 1,800 per year of age, minus 6,500 per kilometre from the
centre, 12,000 for a garage. Noise of 18,000 euros on each price.

Say why this matters: without the truth, "the coefficient is 7,503" is a number.
With it, "the coefficient is 7,503 and the truth is 9,000" is a lesson about
when estimates can be trusted - which is section 7 of the handout and the most
useful thing in today's lesson.
:::

# What we are trying to predict

![](price_vs_area.png)

::: notes
From notebook 1. Before any model, look at the data - the habit lesson 2 spent
three hours arguing for.

Two things to point at. The cloud clearly slopes, so a line is a reasonable
first guess; and the spread around that slope is wide, which is the 18,000 euros
of noise we put in. That spread is a floor: no model, however clever, can
predict these prices better than the noise allows.

Ask the room to eyeball the slope in euros per square metre before you show any
fit. Someone usually lands near 2,400, and having guessed it makes the fitted
number mean something when it arrives.
:::

# What a linear model claims

Each feature contributes a fixed amount per unit, and the contributions add up:
ŷ = w₁x₁ + w₂x₂ + … + wₙxₙ + b, one coefficient w per feature plus an intercept b.


- Every square metre: the same 2,400 €
- Every kilometre out: the same −6,500 €

::: notes
Read it as a sentence, then question it. Is the hundred-and-first square metre
really worth what the fiftieth was? Is a garage worth the same in the centre as
in the suburbs, where there is street parking anyway?

Almost certainly not. The right response is not to abandon linear models but to
know what you have assumed - section 5 shows how to check the assumption and how
to relax it when it fails.

Ask the room for a feature where the constant-per-unit claim would obviously
break. Bedrooms is a good answer: the tenth bedroom is not worth what the second
was.
:::

# Fitting means minimising something

We need one number saying how badly a candidate model is doing.

Adding up the errors, Σᵢ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾), fails: +50,000 and −50,000
cancel to zero.

Every error must count as a positive amount:

- absolute error Σᵢ |ŷ⁽ⁱ⁾ − y⁽ⁱ⁾|
- squared error Σᵢ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²

::: notes
This is the intuition before the formula, and it takes ten seconds. Read the
first sum aloud: for every house i, the prediction minus the true price, added
up. A model that is 50,000 too high on one house and 50,000 too low on another
is not perfect, but that sum is zero.

So every error must count as a positive amount. That is the requirement, and the
last line gives the two ways of meeting it: take the absolute value of each
error, or square it. The next slide says why we choose squaring, and the one
after turns the squared sum into the cost the lesson minimises.
:::

# Why squared, not absolute

- Squared error Σᵢ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²: each error e = ŷ⁽ⁱ⁾ − y⁽ⁱ⁾
  counts as e², not as |⁠e⁠|
- **Differentiable everywhere**: e² is smooth, |⁠e⁠| has a corner at e = 0
- **Punishes one large error more**: double an error and |⁠e⁠| doubles,
  e² quadruples
- Under Gaussian noise it is **maximum likelihood**: it picks the
  coefficients that make the observed prices most probable

::: notes
Start from the formula on the first line, the squared error of the previous
slide: for each house, the prediction minus the true price, squared, then added
up. Call one error e: squared error adds e squared for it, absolute error would
add the size of e.

Then give the three reasons on the slide in increasing order of depth. The first,
differentiability, is convenience: e squared has a derivative everywhere, while
the absolute value has a corner at zero, so with squared error calculus can find
the minimum directly. The second is a design decision: doubling an error
quadruples its cost, so being wrong by 40,000 on one house costs the same as
20,000 on four, which matches most intuitions about what a bad prediction is.

The third is the real reason and worth saying even though we do not prove it: if
you believe the noise is Gaussian, least squares is not a convention, it is the
maximum likelihood estimator - the coefficients it picks are the ones under which
the observed prices were most probable. That connects to the histogram of the
residuals (the leftover errors) in notebook 1, which has the bell shape of a
Gaussian.

Then the consequence, which is not optional: squared error is sensitive to
outliers, because one absurd value contributes its error squared. Lesson 2's work on outliers is needed for this lesson, not an optional warm-up.
:::

# The cost function: mean squared error

$$J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

::: notes
Name the parts: m is the number of examples, the sum runs over all of them,
the half is there for convenience when we differentiate, and the square makes
every error positive and punishes large ones most, as the previous slide said. Strictly,
J is half the mean squared error: the MSE is the same sum divided by m, with no
half. MSE from here on, and the name is worth saying aloud because every later
slide uses the abbreviation.

Ask where the half comes from if nobody has. Answer: differentiating a square
brings down a 2, and the half cancels it. It cannot move the minimum, because
scaling a function by a constant does not move its minimiser.

Handout section 2.2. The worked table there computes the cost for three houses
and shows how the two large errors contribute nearly everything - squared error
spends its attention on the worst predictions, which is a choice you are making
whether or not you notice.
:::

# Trying ŷ = 2,400 · area + 45,000

| Area (m²) | True (€) | Predicted (€) | Error ŷ − y | Error² |
|---|---|---|---|---|
| 80 | 240,000 | 237,000 | −3,000 | 9.0 × 10⁶ |
| 120 | 320,000 | 333,000 | +13,000 | 1.69 × 10⁸ |
| 200 | 540,000 | 525,000 | −15,000 | 2.25 × 10⁸ |

::: notes
Walk the first row across out loud: 80 square metres, the model says 237,000,
the truth is 240,000, so it is 3,000 euros light. Then the last column squares
it.

Dividing the sum by 2m gives a cost of about 6.7 × 10⁷. Nobody should be
impressed by that number on its own - a cost is only meaningful compared with
another cost, which is exactly what fitting does.

Handout section 2.3 has this table. The next slide is what it is really for.
:::

# Squared error spends its attention on the worst house

- The 3,000 € error contributes about **2%** of the total
- The two large errors contribute the other **98%**
- Halving the 15,000 € error cuts the cost by **42%**
- In code: `mean_squared_error(y, y_pred)`, the mean of the squared errors
  without the ½

::: notes
This is the point of the worked example, and it is a design decision rather
than a fact of nature.

Put the third bullet against its opposite, because the contrast is the whole
lesson: halving the *worst* error cuts the cost by 42%, while halving the *best*
one cuts it by 1.7%. Same effort, twenty-five times the reward - so that is
where the fit will go.

Squaring means the fit will bend itself towards its worst predictions and
largely ignore the ones it already gets nearly right. Most of the time that is
what you want - a model wrong by 40,000 on one house is worse than one wrong by
10,000 on four.

But say the flip side plainly, because it is lesson 2 arriving from a new
direction: one absurd outlier contributes its error squared, so it can drag the
entire fit towards itself. If you did not clean the data, least squares will
faithfully optimise for the mistake.

Ask what would change if we minimised absolute error instead. Answer: the fit would follow the bulk of the data and pay much less attention to
the outlier - which is what robust regression (a family of methods built to
resist outliers) does, and why it exists.
:::

# A cost needs something to be compared against

- Trivial answer: **always predict the mean**, `DummyRegressor()`
- Root mean squared error, RMSE = √MSE, on the 150 test houses: **96,440 €**
  for the mean, **20,341 €** for the model
- R² = 1 − Σ(y − ŷ)² ÷ Σ(y − ȳ)² = **0.956** (`r2_score`): 0 is the mean,
  1 is perfect
- Quote the euros too: R² depends on *this* test set's variance

::: notes
Lesson 1's habit, applied to regression for the first time. A classifier gets a majority-class baseline before its score is taken seriously; a regression
gets the mean.

Read the two root mean squared error (RMSE) numbers aloud before the ratio:
96,440 euros against 20,341.
Then say that R-squared is nothing more than that comparison written as one
number - one minus the model's squared error over the mean's - so 0 means you
matched the mean, 1 means perfect, and negative means worse than the mean, which
does happen on a test set.

The last bullet is the caution that keeps them out of trouble, and the handout
measures it: keep this exact model and score it on the 55 test houses whose price
is within half a standard deviation of the mean - a street where the houses
resemble each other. R-squared falls from 0.956 to 0.448 while the RMSE IMPROVES,
20,341 euros to 18,661, because predicting the mean is a good strategy when every
house is worth about the same.

So R-squared compares a model against the mean OF THE DATASET IT WAS MEASURED ON,
and cannot compare two models measured on different data. Handout section 2.4.
:::

# Fitting is finding where J is smallest

$$(w^*, b^*) = \arg\min_{w,\,b}\; J(w, b)$$

::: notes
Read it aloud: w star and b star are the values of the slope and the intercept
at which the cost J is smallest. The star marks "the best values"; it is a
label, not an operation.

arg min is lesson 1's notation, met there as f-hat equals the arg min of the
training error. It returns where the minimum is reached, not the minimum
itself: here, a pair of numbers (w, b), not a cost. That distinction is worth a
sentence, because the minimum of J is a number nobody needs - what the lesson
wants is the coefficients that achieve it.

This line is the whole of "fitting". Everything else in the first half of the
lesson is a way of computing it: exactly, with the normal equation on slides
15-19, or step by step, with gradient descent on slides 22-27. The next slide
shows J as a surface over (w, b), and the pair this formula asks for is the
lowest point of that surface.

Handout section 2.2.
:::

# The cost is a bowl

![](cost_surface.png)

::: notes
From notebook 1. Two parameters, so the cost is a surface over the plane of
(slope, intercept), and fitting means finding its lowest point. The gold star
is the previous slide's (w*, b*): the notebook computes the cost at every
point of a grid and keeps the lowest, a slope of 2,790 and an intercept of
-52,000. The normal equation, three slides on, leads to the exact answer, 2,785
and -51,025, within one grid step of the star.

The important structural fact: this bowl is convex. There is one minimum and no
local traps. Say that explicitly, because it is a property most methods later in
the course do NOT have - lesson 9's networks have cost surfaces full of local
minima, and a great deal of the difficulty of training them comes from exactly
that.
:::

# The picture behind the exact solution

![](projection_picture.png)

::: notes
Spend a full minute here. This is the intuition; the algebra follows.

Each feature is a direction the model can move in. Everything the model can
predict lies on the flat surface those directions span. The true prices are a
point that almost certainly does NOT lie on it - no combination of area and age
reproduces them exactly.

So the best you can do is the closest point on the surface: the shadow, the
perpendicular projection.

Then the punchline, which is the whole method: why perpendicular? Because if the
leftover error had any component lying ALONG the surface, you could have moved
that way and done better. At the optimum the error must be at right angles to
every feature. That is what the normal equations say, and it is where the name
comes from.
:::

# θ and X: all parameters, all houses

- θ = (b, w₁, …, wₙ)ᵀ: every parameter in one column, the intercept first
- X: one row per house; a column of 1s, then one column per feature
- So Xθ gives b + w₁x₁ + … + wₙxₙ for every house at once: all m
  predictions in one product
- The cost becomes J(θ), and the next slide solves for θ

::: notes
Two pieces of notation, and the next slide is unreadable without them.

Theta is a column vector: b on top, then w1 down to wn. It is the course's
symbol for all the parameters together, intercept included. Write it on the
board as a column - b, w1, dots, wn - because that is how it multiplies.

X is the table of features with one extra column of 1s in front. Ask the room
why the 1s are there before you say it: multiply a row of X by theta, and the 1
picks up b while each feature picks up its own w. So one row times theta is
exactly the linear model of slide 6, b + w1 x1 + ... + wn xn, and X times theta
is that prediction for every house at once.

With the three houses of the slide after next: X has rows (1, 80), (1, 120),
(1, 200), and theta is (b, w).

Handout section 3.1 sets this up before the derivation.
:::

# The normal equation

$$\theta = (X^\top X)^{-1}X^\top y$$

::: notes
This is the exact solution: theta collects the intercept and every coefficient,
and the formula gives all of them at once, with no iteration. It is four lines of
NumPy, and notebook 1 shows it agreeing with scikit-learn's answer to within
1.8 × 10⁻¹² euros per square metre - the two differ only by rounding.

Handout section 3.1 derives it properly and section 3.2 does it by hand on three
houses: X transpose X, determinant 22,400, the inverse, and out comes 2,536
euros per square metre against a true 2,400. Three data points, two parameters,
6% error. Worth doing on the board if the room is engaged.

Why it is the minimum and not some other flat point is the next slide but one.
:::

# The same equation, by hand

| Quantity | Value |
|---|---|
| the three houses | 80, 120, 200 m² → 240k, 320k, 540k |
| X, and θ | a column of 1s (for b) beside the areas; θ = (b, w) |
| XᵀX | rows (3, 400) and (400, 60,800) |
| Xᵀy | (1,100, 165,600) |
| determinant | 22,400 |
| **slope w** | **2,536 €/m²** (truth 2,400) |
| intercept b | 28,600 € |

::: notes
Worth doing on the board if the room is awake - it takes four minutes and it
turns the formula into something they have touched.

The whole method on three data points: build the two-by-two matrix, take its
determinant, invert it, multiply. Out comes a slope within 6% of the value that
generated the data, from three houses and two parameters.

If you do put it on the board, show the dot product that gives the second entry
of X transpose y - 80 times 240, plus 120 times 320, plus 200 times 540. It is
the step where an arithmetic slip hides, and watching it done is what makes the
formula stop being a black box.

Then the honest caveat, and it is sharper than it looks: three points is
nothing, and the 6% is luck. Run the SAME one-feature fit on 450 houses and you
get 2,785 - sixteen percent high, further from the truth, with 150 times the
data. More rows cannot fix it, because nothing is broken: asked what a square
metre is worth when area is all you are told, the honest answer includes the
bedrooms and bathrooms that come with it. All six features give 2,421.

So the point is not the accuracy. It is that the formula is arithmetic you could
do by hand - and that what a coefficient means depends on what else is in the
model, which is section 7.

Handout section 3.2 carries every intermediate step.
:::

# And it really is the minimum

- The second derivative is XᵀX, never negative in any direction
- So the cost is **convex**: one bottom, no local traps
- Any stationary point (zero gradient) is *the* answer, not *an* answer
- In code: `LinearRegression().fit(X, y)` solves the same problem

::: notes
Short slide, big consequence. Setting a gradient to zero finds a flat spot; it
does not by itself tell you the flat spot is the lowest point rather than a
ridge or a local dip.

Here it does, because the cost is convex. Give the one-line reason without the
algebra: XᵀX can never be negative in any direction, since asking for its
value in direction v gives the squared length of Xv, and a squared length is
never negative.

Now flag it forward, because this is the slide they should remember in lesson 9:
this property is a luxury. Neural network cost surfaces are full of local
minima, and a large part of the difficulty of training them is that nobody can
promise the point you reached is the point you wanted. Today we can.

Handout section 3.1, the paragraph headed "Is this really a minimum?".
:::

# So why learn anything else?

- $(X^\top X)^{-1}$ costs about $n^3$ operations
- Two columns carrying one fact → **no inverse exists**
- Nearly-redundant columns → an inverse that is useless

::: notes
Three failure modes, and the third is the dangerous one because nothing visibly
breaks.

At a thousand features the cube is a billion operations; at a hundred thousand
it is out of reach. Every large model in this course is trained iteratively.

For the second: notebook 3 constructs exactly this, an area column in metres and
the same in feet.

For the third, give the number from notebook 3: a coefficient that moves from
9,261 to 45,307 across four random splits of the same dataset. The matrix was invertible every time. The coefficients were meaningless every
time.
:::

# How stretched is the valley?

The **condition number** κ = steepest curvature ÷ shallowest curvature: how many
times steeper the steepest direction of the bowl is than the shallowest. The
curvatures are the eigenvalues of XᵀX, so κ = (`np.linalg.cond(X)`)².

- Housing features as recorded: **428,000**
- The same features, standardised: **11.4**
- Add `area_sqft` beside `area_sqm`: **5.2 million**

::: notes
This is the number behind "nearly redundant", and one of the ideas from the
first half of the lesson that is most useful outside it.

Where the curvatures come from, in one sentence: the second derivative of the
cost is the matrix XᵀX divided by m, and its eigenvalues are the curvatures of
the bowl along its steepest and shallowest directions. Warn about the square:
np.linalg.cond(X) gives 654 on the raw housing data, which is the square root of
κ. Read as κ itself, it would understate the stretch more than six hundred times.

The picture: a round bowl is easy to walk down because every direction behaves
the same. Two nearly-identical columns produce a ravine instead - steep across,
almost flat along. The flat direction is the one where moving one coefficient up
and the other down changes the predictions barely at all, which is precisely why
the fit cannot decide between them.

Then the consequence, which appears twice. For the exact solution, a stretched valley
is what makes the inverse enormous and the answer unstable. For gradient descent,
which is the next section, one step size has to serve every direction at once, so the
steep direction sets the limit and the shallow one crawls. The condition number
is, near enough, what sets the iteration count. The handout checks that by
running gradient descent to within 0.1% of the exact solution: 2.9 million
iterations on the raw features, 56 on the standardised ones.

Lesson 2 introduced this for scaling; unscaled features stretch the valley for
exactly the same reason. Handout section 4.4.
:::

# The picture behind gradient descent

You are on a hillside in fog.

You cannot see the valley, but you can feel which way the ground slopes.

So: step against the slope, by an amount set by the **learning rate** α, and
repeat until the cost stops falling.


::: notes
The intuition before any notation. Take a step downhill, feel again, repeat.
That is the entire algorithm.

The gradient is the direction of steepest ascent, so we walk against it. The
learning rate is the length of the stride.

And the danger is equally intuitive, which is the point of using this picture:
stride too far and you cross the valley and end up higher on the far slope.
Everyone has done that walking down a steep hill.
:::

# The gradients: one for w, one for b

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_i \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)}, \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_i \left(\hat{y}^{(i)} - y^{(i)}\right)$$

::: notes
Two parameters, so two slopes to feel in the fog: how fast J rises as w grows,
and how fast it rises as b grows. Both come from differentiating the mean
squared error; handout section 4.1 does it in two lines. Write the error of
house i as e(i) = y-hat(i) - y(i), and both gradients are averages of errors.

Read the first one aloud as a sentence, because its shape carries a lesson: each
example pulls the coefficient in proportion to two things - how wrong the
prediction was, AND how large that feature was for that example.

So a 200 square metre house that is badly mispriced moves the area coefficient
much more than a 40 square metre one with the same error. The gradient is a
weighted vote and the weights are the feature values.

That is precisely why unscaled features cause trouble, and it is lesson 2's
argument arriving from a different direction. The intercept's gradient, on the
right, has no x in it - every example gets an equal vote there, as if b were the
coefficient of a feature that is always 1.

With several features there is one gradient like the first for each w_j, with
x_j in place of x. The next slide puts the two gradients to work.
:::

# One iteration, then the next

1. Predict with the current w and b: ŷ⁽ⁱ⁾ = w·x⁽ⁱ⁾ + b
2. Measure every error: e⁽ⁱ⁾ = ŷ⁽ⁱ⁾ − y⁽ⁱ⁾
3. Average them into the two gradients of the previous slide
4. Step both parameters: w ← w − α·∂J/∂w and b ← b − α·∂J/∂b
5. Back to 1: new w and b give new predictions, so new errors

::: notes
This is the loop the fog picture describes, written as the five things the
computer does, in order. Walk it once with the room before the numbers on the
next slide.

Step 1 uses whatever w and b the previous round left; the very first round
starts from a guess, here w = 0 and b = 0. Steps 2 and 3 turn the predictions
into the two gradients. Step 4 moves BOTH parameters at once, each against its
own gradient, by the same learning rate alpha - and both updates use the
gradients computed from the old values, before either has moved.

Step 5 is the part that makes it iterative, and the part students most often
miss: after the update the model is a different line, so every prediction
changes, so every error changes, so the gradients have to be computed again
from scratch. Nothing from the previous round is reused except w and b
themselves. The loop stops after a fixed number of rounds, or when the cost
stops falling.

In compact form, with theta holding b and w, steps 3 and 4 are the handout's
theta <- theta - alpha times the gradient of J. In notebook 1 the loop is the
`gradient_descent` function, and its body is these steps line for line:
error = (w * x + b) - y, then grad_w = (error * x).mean() and
grad_b = error.mean(), then the two updates. Notebook 1 runs it 4,000 times.

The next slide is one pass through steps 1 to 4, on two houses.
:::

# One step on two houses, from w = 0 and b = 0

| 80 m² at 240k, 200 m² at 540k; α = 10⁻⁵ | Gradient | After one step |
|---|---|---|
| slope w | −63,600 | 0.636 |
| intercept b | −390 | 0.0039 |

::: notes
This is steps 1 to 4 of the previous slide, done once. Say first that the
prices here are in thousands of euros, so 240 means 240,000, which keeps the
numbers small enough to read.

Both gradients are negative - the model starts at zero and predicts zero for
every house, so every prediction is far too low - and both parameters therefore
increase. So far, nothing surprising.

Now the thing to notice, and give the room a moment to spot it: after one
identical step the slope has moved **163 times further** than the intercept.

Nothing clever caused that. In the gradient, each house's error is multiplied
by its area for the slope, and by 1 for the intercept (the intercept behaves like
a feature that is always 1). The two areas are 80 and 200, so the slope's
gradient comes out about 163 times larger before anything else happens. With one shared learning rate, the intercept crawls towards its
value long after the slope has arrived - which is exactly what notebook 1 shows
after 4,000 iterations.

That is the concrete cost of unscaled features, and the cure is lesson 2's.
Handout section 4.2.
:::

# The path it takes

![](gradient_descent_path.png)

::: notes
Both panels are from notebook 1, and they are the same run seen two ways.

Left: the cost against the iteration number. It falls steeply in the first few dozen steps and then flattens into a long tail that is still creeping at 4,000. Point
at the flat part and say what it means - the model was essentially fitted very
early, and almost all the iterations bought almost nothing.

Right: the same run drawn on the contours, with the star marking the exact
solution from the normal equation. The path arrives quickly in the steep
direction and then crawls along the floor of the valley towards the star. That
crawl is the previous slide's 163-to-1 asymmetry, and it is the condition number
made visible.

The two panels are one point: progress stalls because one stride length has to
serve a direction that is steep and a direction that is nearly flat.

Ask what would happen with a much larger stride. Answer on the next slide.
:::

# Three learning rates

![](learning_rate_regimes.png)

::: notes
Left: too small, correct but slow - the cost falls every iteration and simply
takes too many of them. Middle: about right. Right: too large, and it climbs out
of the valley.

The threshold is not a rule of thumb. Handout section 4.3 derives it: for a bowl
whose curvature (how quickly the slope changes) is c, gradient descent converges
only when alpha is below 2/c. With several features the bowl has a different
curvature in each direction, and the steepest one sets the limit; on unscaled
data that steepest direction belongs to the feature with the largest spread.

So the safe learning rate is set by your worst-scaled feature, which is another
argument for scaling everything.

Practical recipe to give them: start at 0.01 on scaled features, watch the cost,
divide by three if it rises. Lesson 5 replaces the recipe with a search.
:::

# Notebook 1, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| one feature, 4-5 | `train_test_split` → 450 houses to fit, 150 to test |
| the exact answer, 15-19 | the normal equation in NumPy, then `LinearRegression` → 2,785 €/m², agreeing to 1.82e-12 |
| the iterative answer, 22-27 | NumPy only: gradient descent → the same answer, slowly |
| all six features, 52-53 | `LinearRegression` → area 2,421, bedrooms −16.6% |
| the baseline, 12 | `DummyRegressor(strategy="mean")`, `r2_score` → 96,440 € against 20,341 €, R² 0.956 |

::: notes
The model, the cost, the exact solution and the iterative one, built from
scratch. 25 minutes. Let them work through it themselves.

The two moments worth pausing on together. First, the agreement between their
four-line normal equation and scikit-learn, to within about 10⁻¹² - seeing it on
their own screen convinces more than any assurance from the lecturer. Second, the coefficient table at the
end, which is the setup for the last section of the lesson.

Watch for anyone whose gradient descent diverges. That is the "try this" at the
end of the notebook and it is a good thing to have happen.
:::

# Break

::: notes
15 minutes. Back for curves, overfitting and regularisation.

Check who is stuck in notebook 1 - the second half assumes everyone got the
coefficient table out, and it is much cheaper to fix now.
:::

# What if the relationship bends?

Energy consumption against temperature: heating in the cold, cooling in the heat.

No straight line follows that.

In code: `PolynomialFeatures(degree=2)` adds the squared temperature, t², as a
second column.


::: notes
Set the problem before the solution. Draw the U shape in the air: consumption
high at both ends, minimum somewhere around 18 degrees.

Ask what a straight line would do here. Someone will say it averages through the
middle - right, and it will be too high in the centre and too low at both ends,
wrong in a structural way rather than slightly imprecise.

Notebook 2 has this dataset: 30 days of measurements, of which 21 are used for
training - deliberately few.
:::

# A relationship a line cannot follow

![](energy_curve.png)

::: notes
From notebook 2. Daily energy consumption against outdoor temperature: heating
at the cold end, cooling at the warm end, a minimum somewhere around 18 degrees.

Let them look at it before saying anything. The U is obvious to a human eye in
under a second, which is worth remarking on - the hard part was never seeing the
shape, it is getting a model to represent it.

Note the sample size out loud: 21 training points. Deliberately few, because the
overfitting we are about to produce needs the model to have nearly as many
coefficients as there are observations, and that situation is far more common in
real work than people expect.
:::

# "Linear" means linear in the coefficients

$$\hat{y} = w_1 t + w_2 t^2 + b$$

::: notes
This is the trick, and it surprises people. The model is a curve in t, but it is
still a weighted sum of columns: one column holds t, a second holds t squared,
and the weights w enter only by multiplication and addition. Nothing in the
derivation of least squares required the columns to be the raw inputs - only
that the model is linear in w, which it still is.

So this is the same least squares, the same normal equation, the same code, on a
design matrix with one extra column. PolynomialFeatures builds those columns.

The same trick covers interactions like x1 times x2, logarithms, anything you can
compute. It is why linear models remain useful on relationships that are not
lines.
:::

# The straight line's residuals keep the shape

![](underfit_residuals.png)

::: notes
This is the diagnostic to teach. The U in the residuals says: there is structure
here the model did not take.

Contrast with notebook 1, where the residuals were a formless cloud with a spread
matching the known noise. That is what "the model got everything" looks like.

Give them the habit: after fitting anything, plot the residuals against each
feature. A pattern means the model has missed something.
:::

# What "nothing left" looks like

![](residuals.png)

::: notes
The contrast to the previous slide, and from notebook 1 - the housing fit, where
the model did capture the structure.

A formless cloud, centred on zero, with no pattern as you sweep left to right.
The spread matches the 18,000 euros of noise we know we put in, which is the
best possible outcome: the model has taken everything that was there to take,
and what remains is the part nobody could have predicted.

Put the two pictures side by side in words: a U means structure you missed, a
cloud means you are done. That is the whole diagnostic, and it costs one line of
code after every fit.

Give them the habit explicitly - plot residuals against each feature, every
time. A pattern means the model has missed something.
:::

# So use more flexibility?

![](polynomial_degrees.png)

::: notes
Three fits to the same 21 points. The degree-1 line is too rigid to follow the
U; degree 2 matches the true curve, which is quadratic; degree 12 swings wildly
between the points.

Ask them which they would pick if the dashed true curve were not drawn - because
that is the real situation. The only thing you ever see is the points.

Then note the trap in the right-hand panel: it passes closest to the training
points of the three. If you selected by training error you would pick it every
time.
:::

# Root mean squared error (RMSE), by degree

| Degree | Train RMSE (kWh) | Test RMSE (kWh) |
|---|---|---|
| 1 | 113.3 | 212.1 |
| 3 | 17.5 | **16.9** |
| 9 | 16.2 | 590.1 |
| 12 | 13.6 | 24,655.7 |

::: notes
Say the name in full before anything else, because every later slide uses the
abbreviation: root mean squared error, RMSE, the square root of the mean
squared error. Taking the root puts the number back into the units of the
target - kilowatt-hours here - so it can be compared with the thing being
predicted. An MSE of 285 means nothing to anyone; an RMSE of 16.9 kWh does.

This is the table the whole lesson turns on. Read the columns against each
other, slowly.

Training error falls at every single step, from 113 down to 14, never once
suggesting anything is wrong. Test error bottoms out at degree 3 and then
climbs: 590 at degree 9, and 24,656 at degree 12.

If someone asks why degree 3 wins when the truth is quadratic: the notebook also
fits degree 2, which scores 17.4 against degree 3's 16.9. With nine test days
that difference is noise; the two are equally good.

So the model that fits the training data best is more than a thousand times
worse on data it has not seen - and the training error congratulated us the
whole way.

This is lesson 1's empirical-versus-expected risk with numbers in it: the error
on the data we fitted against the error on data we have not seen. And note how
little it took: 21 training points, and at degree 12 the model has 12
coefficients plus an intercept to place among them.
:::

# The gap is the overfitting

![](train_test_by_degree.png)

::: notes
The previous table as a picture, and the shape is the one to carry out of the
room.

Training error, the lower line, falls monotonically and never once turns up. If
that were the only curve you could see - and on your own data it is - you would
keep adding degrees forever and congratulate yourself the whole way.

Test error is the other line, and it turns. The gap between the two curves is
the thing with a name: that gap *is* overfitting, and it widens without limit.

This is lesson 1's empirical risk versus expected risk, now with numbers and a
picture. Lesson 5 is about measuring the upper curve, the test error, honestly when you
do not have a test set to spend.
:::

# Notebook 2, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| a relationship that bends, 30-31 | `train_test_split` → 21 days to fit, 9 to test |
| linear in the coefficients, 32-34 | `PolynomialFeatures`, `StandardScaler` and `LinearRegression`, chained by `make_pipeline` |
| too much flexibility, 35-37 | `mean_squared_error`, square-rooted → test RMSE 16.9 at degree 3, 24,655.7 at 12 |
| why the wiggles appear, 39 | the fitted coefficients → largest 3,097,038,010 at degree 12 |

::: notes
Fit the curve, push the degree up, and watch the test error turn. 20 minutes. This is the shortest of the three notebooks and the most visual.

Have them run the degree sweep first and stop at the table, then predict which
degree wins before scrolling. Most rooms guess too high - the instinct that more
flexibility is better is exactly what the lesson is dismantling.

The moment to make sure everyone reaches is the coefficient printout at the end,
because the next slide is about that number and it lands far better if they have
just seen it appear on their own screen.

The "try this" reduces the training set from 21 points to 15 and asks which
degree wins then. The answer moves down, which is the first hint that the right
amount of flexibility depends on how much data you have - a theme for lesson 5.
:::

# Why flexibility goes wrong

Largest coefficient, degree 2: **411**

Largest coefficient, degree 12: **3,097,038,010**

::: notes
Here is the mechanism, and it is the setup for everything after the break.

The huge coefficients work in OPPOSITION. With 21 points and 12 coefficients
there are many ways to pass close to every point, and the one least squares
finds is built from enormous terms that nearly cancel: one pushes the curve up,
the next pushes it down by almost the same amount. At each training point the
cancellation is tuned so the curve lands near the point, because that is all
least squares checks. Between the points nothing tunes it, and the curve swings
by thousands.

Then the question that leads the rest of the lesson: if enormous opposing
coefficients are the symptom, what would happen if we charged for coefficient
size?
:::

# Charge for size

$$J_{\text{ridge}} = \text{MSE} + \lambda\sum_j w_j^2 \qquad J_{\text{lasso}} = \text{MSE} + \lambda\sum_j |w_j|$$

::: notes
Two penalties, one difference: squares or absolute values. MSE here is the
mean squared error from the first half of the lesson. Lambda sets the exchange
rate between fitting the data and keeping coefficients small; at lambda = 0
both are ordinary least squares.

Say the naming trap out loud, because they will hit it in five minutes: we
write the penalty strength as lambda, and reserve alpha for the learning rate.
scikit-learn calls the penalty `alpha`. So `Ridge(alpha=0.01)` in code is
lambda = 0.01 on this slide.

Ridge charges the sum of the squared coefficients, Lasso the sum of their
absolute values; the next slide gives the two rules both of them need.
:::

# Two rules that are not optional

- The **intercept is never penalised**: it is not a claim about any feature
- Features **must be scaled first**, or the penalty is arbitrary
- In code: `make_pipeline(StandardScaler(), Ridge(alpha=0.01))`; scikit-learn
  calls λ `alpha`

::: notes
Both of these are things students get wrong, and both produce results that look
fine, which is what makes them worth a slide of their own.

The intercept says where the surface sits, not what any feature is worth.
Shrinking it would drag every prediction towards zero, which for house prices in
euros is a meaningless place to be dragged. scikit-learn handles this for you in
both Ridge and Lasso - but only because someone decided it should, not because
the formula does it by itself.

Scaling is the sharper one. The penalty is a sum over coefficients, so the same
feature measured in metres and in kilometres attracts penalties differing by a
factor of a thousand. Whichever feature happens to be recorded in small units
gets a large coefficient and is therefore punished hardest, entirely by accident
of the unit someone chose.

Say the contrast with lesson 2 explicitly, because it is the point: there,
scaling was about optimisation speed and you could skip it and still be right.
Here it is correctness. Skip it and the model answers a different question.

Handout section 6.1.
:::

# What a penalty buys

| Model | Train RMSE | Test RMSE | max \|w\| |
|---|---|---|---|
| none | 13.6 | 24,655.7 | 3,097,038,010 |
| λ = 0.01 | 18.0 | **22.8** | 365 |
| λ = 1 | 44.5 | 105.2 | 156 |
| λ = 100 | 96.0 | 227.6 | 6 |

::: notes
Read the last column first: three billion down to 365, from a penalty of one
hundredth. And the test error falls from 24,656 to 23, which is the noise floor: the
noise we added to each day has a standard deviation of 22 kWh, and no model can
predict below it. A thousandfold improvement, bought with one number.

Now the training column, which is the honest part: it gets WORSE at every step.
13.6, 18, 45, 96. That is the trade being made deliberately - a worse fit on the
data we have, in exchange for a better fit on data we do not.

And it stops paying. The test error rises from 22.8 at lambda 0.01, to 105 at
lambda 1, to 228 at lambda 100 - a tenfold loss against the best penalty,
because the model is now too rigid to follow the curve.

The table runs lambda from 0.01 to 100, four orders of magnitude, and the test
error changes tenfold across them. That is why lambda cannot be guessed: it has
to be searched for, which lesson 5 does properly.
:::

# The unpenalised fit leaves the picture

![](ridge_degrees.png)

::: notes
The last column of that table, drawn. Give the room a moment on the left panel
before saying anything.

Point at the vertical scale. The unpenalised curve goes off the top and off the
bottom between the training points, and it only comes back where a point is
sitting there to pull it. That is what a coefficient of three billion looks
like from the outside.

Middle panel, lambda one hundredth: on top of the true curve, the dashed line.
You cannot separate them by eye.

Right panel, lambda one hundred: a nearly flat line. It misses the curve
completely - but look at how it misses. It is wrong by a few hundred kilowatt
hours, on a plot where the left panel is wrong by thousands.

That is the asymmetry to take away, and the table hides it: too much penalty is
a mild failure, too little is a catastrophic one. If you have to err, err
towards more penalty. Handout section 6.4.
:::

# Too flexible, too rigid

![](alpha_trade_off.png)

::: notes
The same table as a picture. Both ends are failures, and they fail for opposite
reasons.

This shape has a name - the bias-variance trade-off - and lesson 5 treats it fully, with the tools to find the bottom of that curve
honestly. Today the point is only that the bottom exists and is not where either
extreme is.

If someone asks how to choose lambda: not by looking at this test curve, because
that would be selecting on the test set. Cross-validation, lesson 5.
:::

# Ridge has an exact solution too

$$\theta = (X^\top X + \lambda I)^{-1}X^\top y$$

::: notes
Put this next to the normal equation from before the break and ask what changed.
The answer is: one term. Add lambda down the diagonal of the matrix you invert,
and that is the entire modification.

Everything else is identical - same derivation, same projection idea, same four
lines of NumPy. Ridge is not a different method; it is
least squares with a price on large coefficients.

Two small print points, for anyone who checks it against the cost on the earlier
slide. Written with theta, the formula penalises the intercept too; in practice
the intercept's entry of I is set to zero, or the data are centred first so the
intercept drops out, which is what scikit-learn does. And with the cost written
as MSE plus lambda times the sum of squares, the lambda in this formula is that
lambda times 2m. Handout section 6.2 shows where the factor comes from.

Note that Lasso has no such formula. The absolute value has a corner, so it
cannot be differentiated at zero, and the solution has to be found iteratively.
That inconvenience is inseparable from the property that makes Lasso useful:
the corner is exactly what produces the zeros, as two slides from now show.
:::

# Why that one change fixes it

- Least squares fails when a direction costs nothing: the valley is flat
- The penalty makes **every** direction cost something
- Every eigenvalue of XᵀX shifts up by λ, so none of them can be zero

::: notes
Come back to the ravine picture from before the break. The reason two identical
columns break least squares is that there is a direction in which moving the
coefficients changes the predictions not at all - the valley is perfectly flat
along it, so there is no unique lowest point. Infinitely many answers fit
equally well.

Ridge tilts the floor of that valley. Once coefficient size costs something, moving along that
flat direction is no longer free, the flatness acquires a slope, and a single
lowest point appears.

The algebraic version is the third bullet and it is worth one sentence: adding
lambda to the diagonal shifts every eigenvalue up by lambda, and a matrix is
singular exactly when one of its eigenvalues is zero. So for any lambda above
zero, it cannot be singular. **Ridge always has a unique answer, even where
least squares has none at all.**

Worth saying plainly: Hoerl and Kennard invented ridge in 1970 for this problem,
not for overfitting. The overfitting cure was the side effect.
:::

# The picture behind Ridge vs Lasso

![](ridge_lasso_geometry.png)

::: notes
Intuition before algebra again, and this one is worth the time.

Think of the penalty as a budget: you may spend a fixed total on coefficients
and want the best fit that money buys. Draw what you can afford. For two
coefficients, charging squares makes the affordable region a disc; charging
absolute values makes it a diamond, a square turned to stand on its corners.

The best affordable fit is where the error contours (the rings of equal cost
around the unpenalised answer) first touch the region.

A disc is smooth, so contact happens at a generic point with both coefficients
non-zero. A diamond has corners, its corners lie ON THE AXES, and a corner is a
point where one coefficient is exactly zero. Corners stick out, so they get
touched first.

That is the entire reason Lasso produces zeros and Ridge does not. Not a
tolerance, not a numerical accident - the shape of the budget.
:::

# Ridge shrinks, Lasso selects

![](regularisation_paths.png)

::: notes
From notebook 3. Left: ridge coefficients approach zero smoothly and never
arrive. Right: lasso coefficients hit zero at a finite lambda and stay.

The housing data again, with all six features scaled, and lambda growing from
left to right. Point at the right panel: one by one the lasso lines reach the
zero line and stay on it. The next slide reads off the order.
:::

# The order Lasso drops features

| λ | Features kept |
|---|---|
| 1,000 | all six |
| 10,000 | five: `garage` dropped |
| 20,000 | three: `area_sqm`, `bedrooms`, `bathrooms` |
| 40,000 | one: `area_sqm` |

::: notes
The previous picture, now as numbers. Read down the right column.

The order is not arbitrary and it is worth asking the room to explain it before
you do. `garage` goes first because, once every feature is on a common scale, it
buys the least accuracy per unit of coefficient spent. `area_sqm` survives
longest because it buys the most.

Nobody told Lasso which features mattered. It was given a budget and told to
spend it well, and this ranking fell out of that - which is genuinely
remarkable and is why Lasso is used as a feature selector at all.

Now the warning, and it is the part that gets forgotten in practice: **the
selection depends entirely on lambda**. "Lasso chose these three features" is not
a complete statement - at a different lambda it chooses differently. Presenting
lasso output as an objective ranking of importance is one of the commonest
misuses of it.

Handout section 6.5.
:::

# The case Ridge was invented for

`area_sqm` and `area_sqft`: the same measurement, two units.

Correlation: **0.999999**

::: notes
The situation from the failure-modes slide, made concrete: someone merged two
systems and the same fact arrived twice.

Ask what least squares should do. The honest answer is that it has no basis for
choosing between "all the effect on the metres column" and "all of it on the
feet column" and everything in between - they all fit identically.

So it picks one arbitrarily, and the arbitrariness changes with the sample.
:::

# The coefficients are noise

| Split | area_sqm | area_sqft |
|---|---|---|
| 0 | 9,261 | −640 |
| 1 | 14,535 | −1,127 |
| 2 | 45,307 | −3,989 |
| 3 | 39,061 | −3,402 |

::: notes
Same dataset, four different random splits. The area coefficient varies by a
factor of five, and its duplicate swings the opposite way to compensate.

Stress what is and is not broken: the PREDICTIONS are fine throughout. What a
square metre adds is the metres coefficient plus 10.76 times the feet
coefficient, since a square metre is 10.76 square feet, and on every split that
sum stays within about 2% of the true 2,400 (the notebook prints 2,424 for its
own split). It is the EXPLANATION that is noise.

Anyone reading these coefficients to learn what drives house prices would draw
nonsense, confidently. With ridge at lambda 10 the same four splits give 38,000
to 41,000 on both columns, split evenly. Say the units, or the comparison
misleads: ridge runs on scaled features, so these are euros per standard
deviation of area, not euros per square metre, and they cannot be set against
the table. What matters is that they are stable and equal, because two moderate coefficients cost
less under a squared penalty than one huge positive and one huge negative.

So ridge is not only a cure for overfitting. It is what makes coefficients
readable when features overlap - which in real data they always do.
:::

# What a coefficient actually says

The change in the prediction for a one-unit change in that feature, **with
every other feature held fixed**.

::: notes
That last clause is doing enormous work and is almost always skipped over. Say
it slowly, then make it concrete.

The coefficient on `bedrooms` is not "what a bedroom is worth". It is what a
bedroom is worth *to a house whose floor area does not change* - which is a
strange object. A house that gains a bedroom without gaining any space has had
its existing rooms subdivided, and that is a genuinely different proposition
from an extra bedroom in an extra 15 square metres.

Ask the room what the coefficient on bedrooms ought to be for the subdivision
version. Plausibly near zero, or negative. The estimate we get, 7,503, is a
blend of that and something else - which is the setup for the next slide.

This is the reason two honest analysts fit the same data, report different
coefficients, and are both right: they included different features, so their
"held fixed" clauses differ. Handout section 7.1.
:::

# When can you trust a coefficient?

![](coefficient_trust.png)

::: notes
From notebook 1, and this is the slide of the lesson most useful outside it.

The x axis is how correlated each feature is with area; the y axis is how far its
estimated coefficient landed from the truth. Read it from left to right. Three
features are uncorrelated with area. Two of them, distance and age, are
recovered to within 1%. The third, garage, is 9.6% out: it is a yes-or-no column,
so it varies little, and the noise weighs more on its coefficient. Then the
correlated ones: bathrooms, correlated 0.49 with area, is 9% out; bedrooms,
correlated 0.77, is 17% out.

State the rule: a coefficient is only as trustworthy as its feature is
independent. And on a real dataset the same wobble is there with nothing to
reveal it - you would just have a number.

Then the harder point, briefly: a coefficient is an association, not a causal
effect. If houses near the centre are older and dearer, age and distance divide
the effect between them however the sample suggests. Neither number tells you
what would happen if you moved a house.
:::

# Notebook 3, live

| section, and its slides | you call (→ the number to come back with) |
|---|---|
| Ridge on the degree-12 fit, 40-44 | `Ridge(alpha=0.01)` after `PolynomialFeatures` and `StandardScaler` → largest coefficient 365 |
| regularisation paths, 47-49 | `make_pipeline(StandardScaler(), Ridge(alpha=...))`, the same with `Lasso` → only area left at 40,000 |
| the collinear case, 50-51 | `LinearRegression` against `Ridge(alpha=10)` on four splits → 9,261 to 45,307, then about 40,000 |

::: notes
Ridge on the degree-12 fit, the regularisation paths, and the collinear case.
20 minutes. The collinear section is the one to make sure everyone reaches - it
is the argument they will need whenever someone asks what a model says drives
the outcome.

The "try this" swaps Ridge for Lasso on the redundant columns. Ask them to
predict which way lasso resolves it before running, then to explain why the two
penalties disagree. It is a good check of whether the disc-and-diamond picture
landed.
:::

# What we did today

- A model that claims a fixed amount per unit
- An exact solution (a projection), and when it fails
- An iterative one: walk downhill, mind your stride
- Flexibility helps until it does not
- Charging for size buys generalisation, and readable coefficients
- **365** against billions: what a penalty of 0.01 costs, and buys

::: notes
Connect the lessons. Lesson 2 prepared data honestly; today we fitted something to
it and found that fitting well and explaining well are different goals.

The one sentence to leave them with: the model that fits your training data best
is usually not the one you want.

And the number to leave them with, repeated from slide 42: a penalty of
one hundredth took the largest coefficient from billions to 365 and the test
error from 24,656 to 22.8. One number, chosen well, bought all of that.

Preview lesson 4 in a sentence: same machinery, but the target becomes a
category instead of a number - and that single change turns out to require a
different cost function, for a reason lesson 4 derives.
:::

# Homework: we discuss it on Friday 23 October

`Exercises/03_regression.md`

Fit, regularise, and **explain which coefficients you believe**.

::: notes
Set it explicitly and say when it is discussed. It uses a new dataset,
bike sharing demand, so the exploration is theirs to do.

Point out the two tasks that carry the most marks. Task 4 asks them to justify a
choice of lambda without looking at the test set, which is genuinely awkward
before lesson 5 and is meant to be: the point is to feel the need for
cross-validation before being handed it. Task 6 asks which coefficients they
trust and why, which is section 7 of the handout applied.

As always: at the exam, accuracy alone earns nothing; the reasoning does.
:::
