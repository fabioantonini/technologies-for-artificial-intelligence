---
title: "Lesson 3: Regression"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "9 October 2026"
---

# Before we start

- Exercise 2 was due today
- The pipeline you built is the one we fit models into from now on

::: notes
Collect exercise 2 and say one sentence about what you saw, if you have looked.
The recurring mistake to name, if it appeared: fitting the scaler or the imputer
before splitting. It is the exact thing lesson 2 warned about, and seeing it in
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
Agenda. Flag that this is the first lesson where we fit something, and that
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
of noise we put in. That spread is the floor: no model, however clever, can
predict these prices better than the noise allows.

Ask the room to eyeball the slope in euros per square metre before you show any
fit. Someone usually lands near 2,400, and having guessed it makes the fitted
number mean something when it arrives.
:::

# What a linear model claims

Each feature contributes a fixed amount per unit, and the contributions add up.

- Every square metre: the same 2,400 €
- Every kilometre out: the same −6,500 €

::: notes
Read it as a sentence, then attack it. Is the hundred-and-first square metre
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

Adding up the errors fails: +50,000 and −50,000 cancel to zero.

::: notes
This is the intuition before the formula, and it takes ten seconds. A model that
is 50,000 too high on one house and 50,000 too low on another is not perfect,
but the sum of its errors is zero.

So every error must count as a positive amount. That is the requirement; squaring
is one way to meet it, absolute value is another. The next slide says why we
choose squaring.
:::

# Why squared, not absolute

- **Differentiable everywhere**: no corner at zero
- **Punishes one large error more** than several small ones
- Under Gaussian noise, it **is** maximum likelihood

::: notes
Give the three in increasing order of depth. The first is convenience. The
second is a design decision: being wrong by 40,000 on one house costs the same
as 20,000 on four, which matches most intuitions about what a bad prediction is.

The third is the real reason and worth saying even though we do not prove it: if
you believe the noise is Gaussian, least squares is not a convention, it is the
maximum likelihood estimator. That connects to the residual histogram in
notebook 1.

Then the consequence, which is not optional: squared error is sensitive to
outliers, because one absurd value contributes its error squared. Lesson 2's
outlier work is a prerequisite for this lesson, not a preliminary.
:::

# The cost function: mean squared error

$$J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

::: notes
Name the parts: m examples, the half for convenience when we differentiate,
the square doing the work discussed. This is the mean squared error, MSE from
here on, and the name is worth saying aloud because every later slide uses the
abbreviation.

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

Ask what would change if we minimised absolute error instead. Answer: the fit
would follow the bulk of the data and shrug at the outlier - which is what
robust regression does, and why it exists.
:::

# The cost is a bowl

![](cost_surface.png)

::: notes
From notebook 1. Two parameters, so the cost is a surface over the plane of
(slope, intercept), and fitting means finding its lowest point.

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

# The normal equation

$$\theta = (X^\top X)^{-1}X^\top y$$

::: notes
Four lines of NumPy, and notebook 1 shows it agreeing with scikit-learn to
1.8e-12 euros per square metre.

Handout section 3.1 derives it properly and section 3.2 does it by hand on three
houses: X transpose X, determinant 22,400, the inverse, and out comes 2,536
euros per square metre against a true 2,400. Three data points, two parameters,
6% error. Worth doing on the board if the room is engaged.

Also mention the second derivative is X transpose X, which is positive
semi-definite, so the cost is convex and any stationary point is global. That is
the formal version of "the bowl has one bottom".
:::

# The same equation, by hand

| Quantity | Value |
|---|---|
| the three houses | 80, 120, 200 m² → 240k, 320k, 540k |
| Xᵀ X | rows (3, 400) and (400, 60,800) |
| Xᵀ y | (1,100 , 165,600) |
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

Then the honest caveat: three points is nothing, and 6% here is luck as much as
method. Notebook 1 does the same on 450 houses and gets 2,410. The point is not
the accuracy, it is that the formula is arithmetic you could do by hand.

Handout section 3.2 carries every intermediate step.
:::

# And it really is the minimum

- The second derivative is Xᵀ X, never negative in any direction
- So the cost is **convex**: one bottom, no local traps
- Any stationary point is *the* answer, not *an* answer

::: notes
Short slide, big consequence. Setting a gradient to zero finds a flat spot; it
does not by itself tell you the flat spot is the lowest point rather than a
ridge or a local dip.

Here it does, because the cost is convex. Give the one-line reason without the
algebra: Xᵀ X can never be negative in any direction, since asking for its
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
9,261 to 45,307 across four random splits of the same dataset. The matrix was
invertible every time. The answer was garbage every time.
:::

# How stretched is the valley?

The **condition number** compares the steepest direction with the shallowest.

- Housing features as recorded: **285**
- The same features, standardised: **3.4**
- Add `area_sqft` beside `area_sqm`: **2,286**

::: notes
This is the number behind "nearly redundant", and it is the most transferable
idea in the first half of the lesson.

The picture: a round bowl is easy to walk down because every direction behaves
the same. Two nearly-identical columns produce a ravine instead - steep across,
almost flat along. The flat direction is the one where moving one coefficient up
and the other down changes the predictions barely at all, which is precisely why
the fit cannot decide between them.

Then the payoff, which lands twice. For the exact solution, a stretched valley
is what makes the inverse enormous and the answer unstable. For gradient descent
 -  next section - one step size has to serve every direction at once, so the
steep direction sets the limit and the shallow one crawls. The condition number
is, near enough, what sets the iteration count.

Lesson 2 introduced this for scaling; unscaled features stretch the valley for
exactly the same reason. Handout section 4.4.
:::

# The picture behind gradient descent

You are on a hillside in fog.

You cannot see the valley, but you can feel which way the ground slopes.

::: notes
The intuition before any notation. Take a step downhill, feel again, repeat.
That is the entire algorithm.

The gradient is the direction of steepest ascent, so we walk against it. The
learning rate is the length of the stride.

And the danger is equally intuitive, which is the point of using this picture:
stride too far and you cross the valley and end up higher on the far slope.
Everyone has done that walking down a steep hill.
:::

# The update rule

$$w \leftarrow w - \alpha\,\frac{\partial J}{\partial w}, \qquad \frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_i \left(\hat{y}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

::: notes
Read the gradient aloud as a sentence, because its shape carries a lesson: each
example pulls the coefficient in proportion to two things - how wrong the
prediction was, AND how large that feature was for that example.

So a 200 square metre house that is badly mispriced moves the area coefficient
much more than a 40 square metre one with the same error. The gradient is a
weighted vote and the weights are the feature values.

That is precisely why unscaled features cause trouble, and it is lesson 2's
argument arriving from a different direction. The intercept gradient has no x in
it - every example gets an equal vote there.

Handout 4.2 works one step by hand: after a single update the slope has moved
163 times further than the intercept.
:::

# One step, from w = 0 and b = 0

| α = 10⁻⁵ | Gradient | After one step |
|---|---|---|
| slope w | −63,600 | 0.636 |
| intercept b | −390 | 0.0039 |

::: notes
Both gradients are negative - the model predicts zero for everything, so every
prediction is far too low - and both parameters therefore increase. So far, so
unremarkable.

Now the thing to notice, and give the room a moment to spot it: after one
identical step the slope has moved **163 times further** than the intercept.

Nothing clever caused that. Areas are around 140, and the intercept's "feature"
is always 1, so the slope's gradient is roughly 140 times larger before anything
else happens. With one shared learning rate, the intercept crawls towards its
value long after the slope has arrived - which is exactly what notebook 1 shows
after 4,000 iterations.

That is the concrete cost of unscaled features, and the cure is lesson 2's.
Handout section 4.2.
:::

# The path it takes

![](gradient_descent_path.png)

::: notes
Both panels are from notebook 1, and they are the same run seen two ways.

Left: the cost against iteration. It drops off a cliff in the first few dozen
steps and then flattens into a long tail that is still creeping at 4,000. Point
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

The threshold is not folklore. Lesson 2 derived it: for a bowl of curvature c,
convergence requires alpha below 2/c, and with several features the binding
constraint is the largest curvature - the largest feature variance.

So the safe learning rate is set by your worst-scaled feature, which is another
argument for scaling everything.

Practical recipe to give them: start at 0.01 on scaled features, watch the cost,
divide by three if it rises. Lesson 5 replaces the recipe with a search.
:::

# Notebook 1, live

The model, the cost, the exact solution, the iterative one: from scratch.

::: notes
30 minutes. Let them drive.

The two moments worth pausing on together. First, the agreement between their
four-line normal equation and scikit-learn to twelve decimal places - it lands
better than any assurance from the front. Second, the coefficient table at the
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

::: notes
Set the problem before the solution. Draw the U shape in the air: consumption
high at both ends, minimum somewhere around 18 degrees.

Ask what a straight line would do here. Someone will say it averages through the
middle - right, and it will be too high in the centre and too low at both ends,
wrong in a structural way rather than slightly imprecise.

Notebook 2 has this dataset: 30 days of measurements, which is deliberately few.
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
The trick, and it surprises people. Nothing in the derivation required the
inputs to be straight - only that the model is linear in w, which it still is.

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
feature. A pattern is a message.
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
time. A pattern is a message.
:::

# So use more flexibility?

![](polynomial_degrees.png)

::: notes
Degree 1 too rigid, degree 2 right, degree 12 wild.

Ask them which they would pick if the dashed true curve were not drawn - because
that is the real situation. The only thing you ever see is the points.

Then note the trap in the right-hand panel: it passes closest to the training
points of the three. If you selected by training error you would pick it every
time.
:::

# Root mean squared error (RMSE), by degree

| Degree | Train RMSE | Test RMSE |
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

Training error falls at every single step, 113 down to 16, monotonically, never
once suggesting anything is wrong. Test error bottoms out at degree 3 and then
climbs: 118 at degree 9, 182 at degree 12.

So the model that fits the training data best is ten times worse on data it has
not seen - and the training error congratulated us the whole way.

This is lesson 1's empirical-versus-expected risk with numbers in it. And note
how little it took: 21 training points, and by degree 9 the model has nearly as
many coefficients as data.
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
picture. Lesson 5 is about measuring the upper curve honestly when you do not
have a test set to spend.
:::

# Notebook 2, live

Fit the curve, push the degree up, and watch the test error turn.

::: notes
20 minutes. This is the shortest of the three notebooks and the most visual.

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
finds has terms that nearly cancel - one pushing the curve up where the next
pushes it down, the cancellation failing exactly where a training point sits.
Between the points, nothing constrains the swing.

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

Two things that are not optional and that students get wrong. The intercept is
never penalised - it is not a claim about any feature, and shrinking it would
bias every prediction towards zero. And features MUST be scaled first: the
penalty is a sum over coefficients, so the same feature in metres and in
kilometres attracts penalties differing by a factor of a thousand.

Lesson 2 introduced scaling for optimisation speed. Here it is correctness.
:::

# Two rules that are not optional

- The **intercept is never penalised**: it is not a claim about any feature
- Features **must be scaled first**, or the penalty is arbitrary

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

| Model | Train | Test | max \|w\| |
|---|---|---|---|
| none | 13.6 | 24,655.7 | 3,097,038,010 |
| λ = 0.01 | 18.0 | **22.8** | 365 |
| λ = 1 | 44.5 | 105.2 | 156 |
| λ = 100 | 96.0 | 227.6 | 6 |

::: notes
Read the last column first: three billion down to 365, from a penalty of one
hundredth. And the test error falls from 24,656 to 23, which is the noise floor -
a thousandfold improvement bought with one number.

Now the training column, which is the honest part: it gets WORSE at every step.
16, 18, 45, 96. That is the trade being made deliberately - a worse fit on the
data we have, in exchange for a better fit on data we do not.

And it stops paying. At lambda 100 the test error is 228, worse than no penalty
at all, because the model is now too rigid to follow the curve.

The useful range spans four orders of magnitude, which is why lambda cannot be
guessed.
:::

# Too flexible, too rigid

![](alpha_trade_off.png)

::: notes
The same table as a picture. Both ends are failures, and they fail for opposite
reasons.

This shape has a name - the bias-variance trade-off - and lesson 5 gives it the
treatment it deserves along with the machinery to find the bottom of that curve
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
lines of NumPy. Ridge is not a different method; it is least squares with the
floor tilted.

Note that Lasso has no such formula. The absolute value has a corner, so it
cannot be differentiated at zero, and the solution has to be found iteratively.
That inconvenience is inseparable from the property that makes Lasso useful  - 
the corner is exactly what produces the zeros.

Handout section 6.2 derives it in three lines.
:::

# Why that one change fixes it

- Least squares fails when a direction costs nothing: the valley is flat
- The penalty makes **every** direction cost something
- Every eigenvalue shifts up by λ, so none of them can be zero

::: notes
Come back to the ravine picture from before the break. The reason two identical
columns break least squares is that there is a direction in which moving the
coefficients changes the predictions not at all - the valley is perfectly flat
along it, so there is no unique lowest point. Infinitely many answers fit
equally well.

Ridge tilts the floor. Once coefficient size costs something, moving along that
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
and want the best fit that money buys. Draw what you can afford. Charging
squares gives a disc; charging absolute values gives a diamond on its corners.

The best affordable fit is where the error contours first touch the region.

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

Give the order in which lasso drops features on the housing data, because it is
not arbitrary: garage first at lambda 10,000 - the smallest effect once
everything is on a common scale - then distance and age, and by 40,000 only area
survives.

Nobody told it which features matter. It was told to keep the total size small,
and this is the arrangement that buys the most accuracy per euro of coefficient
spent.

The warning: the selection depends entirely on lambda, so "Lasso chose these
features" is never a complete statement.
:::

# The order Lasso drops features

| λ | Features kept |
|---|---|
| 1,000 | all six |
| 10,000 | five: `garage` dropped |
| 20,000 | three: `area`, `bedrooms`, `bathrooms` |
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
selection depends entirely on lambda**. "Lasso chose these four features" is not
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

Stress what is and is not broken: the PREDICTIONS are fine throughout - the
combined effect is stable at about 2,420, close to the true 2,400. It is the
EXPLANATION that is noise.

Anyone reading these coefficients to learn what drives house prices would draw
nonsense, confidently. With ridge at lambda 10 the same four splits give 38,000
to 41,000 on both columns, split evenly, because two moderate coefficients cost
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
From notebook 1, and this is the most transferable slide of the lesson.

The x axis is how correlated each feature is with area; the y axis is how far its
estimated coefficient landed from the truth. The three features uncorrelated with
area - distance, age, garage - are recovered to within 1%. Bathrooms, at 0.49,
is out by 9%. Bedrooms, at 0.77, is out by 17%.

State the rule: a coefficient is only as trustworthy as its feature is
independent. And on a real dataset the same wobble is there with nothing to
reveal it - you would just have a number.

Then the harder point, briefly: a coefficient is an association, not a causal
effect. If houses near the centre are older and dearer, age and distance divide
the effect between them however the sample suggests. Neither number tells you
what would happen if you moved a house.
:::

# Notebook 3, live

Ridge on the disaster, the regularisation paths, the collinear case.

::: notes
30 minutes. The collinear section is the one to make sure everyone reaches - it
is the argument they will use in their own projects when a stakeholder asks what
the model says drives the outcome.

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
Draw the thread. Lesson 2 prepared data honestly; today we fitted something to
it and found that fitting well and explaining well are different goals.

The one sentence to leave them with: the model that fits your training data best
is usually not the one you want.

And the number to leave them with, repeated from before the break: a penalty of
one hundredth took the largest coefficient from billions to 365 and the test
error from 24,656 to 22.8. One number, chosen well, bought all of that.

Preview lesson 4 in a sentence: same machinery, but the target becomes a
category instead of a number - and that single change turns out to require a
different cost function, for a reason that is one of the exam derivations.
:::

# Homework: due Friday 16 October

`Exercises/03_regression.md`

Fit, regularise, and **explain which coefficients you believe**.

::: notes
Set it explicitly with the deadline. It uses a new dataset - bike sharing demand
 -  so the exploration is theirs to do.

Flag the two tasks that carry the most marks. Task 4 asks them to justify a
choice of lambda without looking at the test set, which is genuinely awkward
before lesson 5 and is meant to be: the point is to feel the need for
cross-validation before being handed it. Task 6 asks which coefficients they
trust and why, which is section 7 of the handout applied.

As always: no marks for accuracy.

Also remind them the project topic must be confirmed by lesson 4 - next week.
:::
