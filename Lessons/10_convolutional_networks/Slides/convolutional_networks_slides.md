---
title: "Lesson 10: Convolutional Networks and Course Synthesis"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "27 November 2026"
---

# Agenda
- A convolution, and the two assumptions inside it
- Equivariance and pooling: how position stops mattering
- One defect, moved half an image
- Transfer learning, and its narrow window
- What ten lessons add up to

::: notes
Frame the three hours before any content. Last week's networks treated the 64
pixels of a digit as 64 unrelated numbers. Today exactly one assumption
changes - that the inputs have an arrangement worth keeping - and the whole
lesson is the consequence of that single change, measured rather than
asserted.

Say out loud that this is the last lesson, so it has two jobs rather than one:
teach the convolution properly, and add the ten weeks up. The final segment is
not a victory lap. It runs every family the course has met on one problem, and
the result is not the one the table was built to show.

Nothing today needs a graphics card, a downloaded model, or a network call.
The largest network in this lesson has 1,537 parameters. Handout section 1.
:::

# Exercise 9 returned

- Marks were for the **spread** reported beside each number, not the number
- Recurring gap: a hand-written backward pass, never gradient-checked
- Conclusions drawn from a single seed, presented as differences
- Today the same obligation lands on an **architecture** claim

::: notes
Hand exercise 9 back briefly. One sentence on what went well, one on the
recurring gap. Two things came up repeatedly: gradients written by hand and
trusted without the four-line check, and single-seed runs whose differences
were smaller than the seed-to-seed spread would have been.

Link it forward deliberately, because today is the easiest lesson in the course
in which to fool yourself. Every claim this afternoon is of the form
"architecture A beats architecture B", and that is precisely the claim a lucky
seed manufactures for free. Every table today therefore carries either a spread
or a ceiling beside it, and one of them carries both.

Ask them to keep one question live all afternoon: against what was this
difference measured?
:::

# The sentence lesson 9 ended on

- "Permute the pixels consistently and nothing here notices"
- True of every model in that lesson, and an admission
- A 24×24 image is **not** 576 unrelated numbers
- Neighbouring pixels are the whole point of a photograph
- Today: what you get by refusing to throw the arrangement away

::: notes
Put last week's closing sentence back on the screen, because this lesson exists
to cash it. A dense layer sees a flat vector of 576 numbers. Shuffle those 576
positions with one fixed permutation, apply it to every image in the dataset,
and a dense network trains to exactly the same accuracy. It never knew which
pixel was next to which.

That is a strange property for a model of a photograph. Adjacency is not
incidental in an image; it is what makes it an image rather than a bag of
brightness values.

Ask the room the obvious follow-up before moving on: if the convolution is the
architecture that *does* use the arrangement, what should happen to it under
that same permutation? Hold their answer. We measure it at 2:43, and most
people in the room will be wrong. Handout section 1.
:::

# One assumption changes

- Same loss, same optimiser, same backpropagation as last week
- One new layer type, and inside it **two** assumptions
- Everything else today is consequence, or measurement
- No new mathematics after the first twenty-five minutes

::: notes
Reassure them about scope. Nothing derived last week is superseded: the chain
rule right to left, the ŷ − y gradient at the output, the initialisation and
optimiser advice all apply unchanged. A convolutional layer is a layer, and it
trains the way every other layer trains.

What is new is the *shape* of the layer, and the two assumptions that shape
encodes. Name them now and keep the pair visible all afternoon, because the
final segment takes them apart: weight sharing says a pattern means the same
thing wherever it appears; locality says nearby pixels belong together. The
usual one-line justification for convolution bundles the two, and one of
today's measurements shows they are not the same assumption at all.

Handout sections 1 and 9.
:::

# Not "convolution wins on images"

- A random forest on raw pixels will come within **0.002** today
- Lesson 9's dense network will lose to plain logistic regression
- Four hand-written numbers put every classical family at the ceiling
- What wins is the **representation**, not the model family

::: notes
Say this at the start rather than at the end, so that today's headline numbers
are read honestly when they arrive. It would be easy to spend three hours
building a case that convolutional networks are simply better at images, and
the final segment of this lesson does not support it.

On the same wafer photographs, a random forest given raw pixels and no notion
whatsoever of which pixel adjoins which reaches 0.9780 against the
convolutional network's 0.9800. Lesson 9's dense network reaches 0.7430,
behind logistic regression's 0.8905. The spread across families is enormous,
and it does not line up with how modern each family is.

The thing that changes everything is what the model is given to look at. That
is the sentence the course closes on, and section 12 is where it is measured.
:::

# Three results that contradict a slogan

- Permuting the pixels costs the convolution **nothing** on detection
- Transfer learning helps in a **window**, and is negative outside it
- Freezing a pre-trained base hurt even the **good** source
- All three are measured this afternoon, in front of you

::: notes
Advertise the three uncomfortable findings now, because each contradicts
something students will have read online, and knowing they are coming makes
them land rather than slide past.

The received versions are: convolutions work because images have spatial
structure; transfer learning helps most when data is scarcest; and freezing the
base of a pre-trained network is the safe default. Each is defensible as a
slogan. Each is wrong or incomplete on this data, and today's numbers say in
which direction.

Tell them explicitly that these were not planned findings. The permutation
experiment was written expecting the convolution to collapse on both tasks; the
freezing comparison was included as a control. Both surprised the person who
wrote them, which is the only reason they are in the lesson.

Handout sections 9 and 10.
:::

# Five numbers so far; today the sixth

- **77%** accuracy on coin-flip labels, from leakage alone
- **94 of 128** imputed rows borrowed from the test set
- **365**: the largest coefficient a 0.01 penalty leaves
- **37 of 40** disguised accounts caught by reconstruction error
- A sigmoid layer divides the gradient by about **four**

::: notes
This course has produced one number per lesson worth carrying out of the room,
and this is the last time the list gets extended. Read the five out - they are
lessons 1, 2, 3, 8 and 9 - and say that the sixth arrives at about half past
one this afternoon.

The point of the list is not nostalgia. Each of these numbers was a surprise to
somebody who had the correct general principle already: leakage is bad,
imputation must sit inside the fold, regularisation controls coefficients,
anomalies are relational, sigmoids attenuate gradients. Knowing the principle
did not protect anyone. Seeing the number did.

Ask them to leave a line free in their notes for today's, and to write down now
what they think it will be about.
:::

# Meridian's wafers

- Lesson 9's optical sensors are cut from silicon wafers
- Every die is photographed at **24×24 pixels** and graded
- Rejected if it carries a defect: a scratch of about seven pixels, or a bright particle
- Film thickness varies smoothly across each die

::: notes
Introduce the problem physically before any picture. Meridian Instruments is
the fictional sensor maker from last week; this week we are one step upstream,
on the wafer line where the sensors are cut. A die is a small square of
silicon, photographed under a microscope at 24 by 24 pixels - 576 numbers - and
graded pass or fail by an automatic station.

The last bullet is the one that decides the mathematics. The optical film laid
over the wafer varies in thickness across the die, so the overall brightness of
a photograph tells you about the coating, not about defects. A scratch is dark
*relative to its immediate surroundings*; it is not dark in absolute terms, and
on a thin-film region a scratch may be brighter than clean silicon elsewhere on
the same die.

Ask them what that rules out before showing the figure: any rule of the form
"reject if pixel brightness falls below a threshold". Handout section 1.1.
:::

# Six clean dies, six scratched

![](wafer_examples.png)

::: notes
Give them a moment to find the defects themselves before pointing at any. Most
people find the scratches within a few seconds, which is worth naming: the
human visual system solves this problem instantly and is doing something a
dense layer cannot do.

Point at two panels with visibly different background brightness and the same
defect. That is the film thickness varying, and it is why absolute brightness
is useless here. Then point at how few pixels the defect actually occupies  - 
about seven of 576, roughly one percent of the image.

Ask the room: how many of these twelve images would you need to see before you
could grade a thirteenth? People answer two or three. Lesson 9's dense network
is still 0.235 short of the ceiling with eight thousand, and the reason why is
the whole first half of this lesson. Handout section 1.1.
:::

# Two properties decide everything

- **Local**: a handful of the 576 pixels matter, the rest is irrelevant
- **Position-independent**: top left and bottom right are the same event
- Both were visible in the twelve photographs
- Every design choice this afternoon follows from these two lines

::: notes
These two sentences are the whole inductive bias of a convolutional network,
stated in the language of the problem rather than the language of the
architecture. Get them agreed in the room before any layer is defined, because
the rest of the lesson is a construction that assumes them.

Locality: the evidence for a defect concerns a few adjacent pixels. Nothing
about a scratch in the top left depends on what the bottom right looks like.

Position-independence: a scratch is graded as a scratch wherever it lands. The
station does not have a special rule for row 5. That is a fact about the
physical process, and it is the fact a dense layer cannot use.

Ask them which of the two properties they think is doing more of the work.
Hold the answers - section 9 measures it, and the answer is not an even split.
:::

# The grading station is wrong 2% of the time

- Published error rate **2%**; notebook 01's batch drew **2.55%**
- A high draw at 2.5 standard deviations, not a bug
- Every score today is read against the ceiling of the set it was measured on
- For the 2,000-image test set that ceiling is **0.9800**

::: notes
Lesson 9 had a test rig wrong 3% of the time and a ceiling of 0.97. Same
discipline, new number: this station mislabels 2% of dies, so no classifier can
be *scored* above 0.98 no matter how good it is.

The 2.55% is worth a sentence of its own because a student who checks will find
it and assume something is broken. It is not: with a few thousand dies, a 2%
process draws 2.55% about as often as you would expect, and 2.5 standard
deviations above the mean is an ordinary event when you look at several
batches. The response is to quote the ceiling realised on the set actually
used, not the design figure.

Put the question to the room: if the station is wrong 2% of the time and a
model is right about every die, what accuracy will you measure? The answer is
on the screen in half an hour, and it is exactly 0.98. Handout section 1.1.
:::

# One neuron per pixel is the wrong shape

- A dense unit on this image has **576 weights**, one per pixel
- A unit that has learned to spot a scratch here...
- ...knows nothing about the same scratch three pixels to the left
- Those are different weights, trained by different examples
- Nothing in the architecture connects them

::: notes
This is the core complaint against a dense layer on an image, and it is not
about size. Take a unit that has successfully learned to fire on a scratch at
rows 4 to 6, columns 10 to 12. Its evidence lives in nine of its 576 weights.
Now move the scratch three columns left. Those nine weights see clean silicon;
nine entirely different weights see the scratch, and nothing has ever taught
them anything.

Say plainly that the dense layer is not *unable* to solve this. Given enough
examples of scratches at every position it will learn a detector at every
position, independently, at full price each time. The complaint is about the
bill, and section 8 prints it.

Ask the room how many independent detectors a 24×24 image would need if each
covers a 3×3 patch. About 576 of them - trained separately. Handout section 2.
:::

# 80 weights against 147,712

| layer | parameters |
|---|---|
| dense, 256 units | 147,712 |
| dense, 1024 units | 590,848 |
| convolutional, 8 kernels of 3×3 | **80** |
| convolutional, 32 kernels of 5×5 | 832 |

::: notes
The eight kernels of 3×3 come to 80 parameters: eight kernels of nine weights,
plus one bias each. The dense layer of 256 units on a 576-pixel image comes to
147,712. The ratio is 1,846.

Let the number land, then immediately warn them against the obvious reading.
Almost every textbook presents this table as a story about efficiency, and
efficiency is the least interesting thing in it. If parameter count were the
point, we could get the same saving by using 256 dense units on a downsampled
image, and that helps nothing at all.

The next two slides say what the saving actually is. Ask them first: where did
those 147,632 missing parameters *go*? They were not deleted. They were tied
together. Handout section 2.
:::

# Compression is the least interesting reading

- The convolutional layer produces **more** numbers, not fewer
- Eight feature maps of 24×24 is **4,608** outputs, against 256
- It is not a smaller layer: it is a differently **tied** one
- The 1,846-fold parameter ratio is a side effect

::: notes
This slide exists because the parameter table misleads almost everybody the
first time. Eighty parameters sounds like a tiny layer. It is not a tiny layer:
it emits 4,608 numbers, eighteen times as many as the 256-unit dense layer it
was compared against.

Make the distinction explicit, because it is the one that generalises. The
convolutional layer has not removed capacity; it has constrained *which*
functions the capacity can express, by forcing the same nine weights to be
reused at every one of the 576 positions. Weights are shared, not discarded.

This is the first appearance of an idea the rest of the lesson depends on: a
restriction that costs expressiveness can buy something worth more. Ask them
what it might buy, and take answers before the next slide. Handout section 2.
:::

# What weight sharing buys is evidence

- The same nine weights are applied at **every** position
- So one scratch, anywhere, teaches the detector **everywhere**
- The dense layer must be taught position by position
- Not a memory saving: a **data** saving
- We price it, in images, at 1:35

::: notes
This is the sentence to make them write down. Weight sharing converts one
labelled example into evidence about every position in the image, because the
gradient from a defect at row 4 updates exactly the same nine numbers that
handle row 20.

Contrast it with the dense layer once more, in terms of data rather than
memory. To learn a scratch detector at 576 positions, a dense network needs
scratches at 576 positions. The convolutional network needs scratches.

The phrase "inductive bias" belongs here, and it is worth defining carefully:
an assumption built into the model's structure rather than learned from data.
Weight sharing is one. It is not free - it is only correct because of the
second property two slides ago - but when it is correct, it is worth more than
any amount of tuning. Handout section 2.
:::

# Before the mathematics: what would you compute?

- Look at the twelve dies again
- What number would you compute from an image to grade it?
- If your answer involves a small window and the word "anywhere"...
- ...you have already derived the next forty minutes

::: notes
Run this properly rather than rhetorically: take three answers from the room
before showing anything, and write them on the board. It takes two minutes and
it changes how the next segment is received, because most rooms produce the
right answer without the vocabulary.

Typical answers, and what to do with each. "The darkest pixel" - good instinct,
but defeated by the film thickness, so ask what it should be compared against.
"The difference between a pixel and its neighbours" - that is the contrast
kernel, and they have just invented section 3.3. "The variance of the image"  - 
close, but a smooth brightness gradient has variance too.

The answer we are heading for is: slide a small window over the image, compute
one contrast number at every position, and take the largest. That is a
convolution followed by a global maximum, which is the whole architecture.
Handout section 1.1.
:::

# The convolution, in words

- Take a small array of weights: the **kernel**
- Slide it over the image, one position at a time
- At each position, write down the weighted sum of the pixels it covers
- The result is a **feature map**: one number per position
- The kernel is small; the image is not

::: notes
Say it in words first and let them picture the sliding window before the
notation arrives. A kernel is a stencil of nine numbers. Lay it over the
top-left 3×3 corner of the image, multiply the nine weights by the nine pixels
underneath, add them up, and write that single number down. Slide one column
right and repeat.

Two consequences worth naming immediately. The output is an image, not a
number - smaller, and often much deeper, but still laid out in space. And the
same nine weights produced every entry of it, which is the weight sharing from
the previous segment made concrete.

Ask the room what the output of the sliding window means at each position. It
is the answer to one fixed question - "how much does the pattern I encode look
like what is here?" - asked everywhere at once. Handout section 3.1.
:::

# One weighted sum, at every position

$$(I * K)[r, c] = \sum_{i=0}^{f-1}\sum_{j=0}^{f-1} I[r + i,\; c + j]\; K[i, j]$$

::: notes
Read it out in words rather than in symbols, since the words were the previous
slide: the entry of the output at row r, column c is the sum over the kernel's
own rows and columns of image pixel times kernel weight, with the kernel's
top-left corner placed at r, c.

Two details to point at. The image index is r + i, so the window moves with r  - 
that is the sliding. And the kernel index is i alone, with no r in it: the same
weight is used at every output position, which is weight sharing written down
in one subscript.

This expression is eleven lines of Python in notebook 01, and the notebook
checks it against scipy's implementation rather than against itself. Handout
section 3.1.
:::

# Strictly, this is cross-correlation

- A true convolution flips the kernel before multiplying
- Every deep learning library computes the formula above and calls it convolution
- The kernel is **learned**, so the flip changes nothing that matters
- Notebook 01 agrees with scipy to $8.9 \times 10^{-16}$: rounding, not error

::: notes
Mention the naming discrepancy once, so that when a student meets it in a
signal-processing course or a textbook they are not derailed by it. In signal
processing, convolution flips the kernel and cross-correlation does not. Every
deep learning library implements the unflipped version and calls it
convolution. This course follows the library.

Say why it genuinely does not matter here: the kernel is learned. Whichever
orientation the operation uses, gradient descent will find the weights that
suit it, and the two conventions reach mirror-image kernels with identical
outputs. It matters only when you write a kernel down by hand, which we do in
ten minutes - and the four kernels we write are all symmetric.

The scipy check is worth a sentence: eleven lines of your own code agreeing
with somebody else's implementation to fifteen decimal places is what
"verified" means. Handout section 3.1.
:::

# How big is the output?

$$\text{output size} = \left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

::: notes
Four symbols: n is the input size along one dimension, f the kernel size, p the
padding added on each side, s the stride, and the formula gives the output size
along that same dimension. Square images make both dimensions the same, which
is why this is quoted as one formula rather than two.

The floor is doing real work rather than tidying up. When the stride does not
divide the available positions evenly, the last window would hang off the edge
of the image, and it is simply not taken - those pixels are never looked at by
that layer. Students who have only ever used `padding="same"` meet this the
first time they set a stride and lose a row.

Get them to predict the next slide's five answers before it appears. Handout
section 3.2.
:::

# Worked on this lesson's 24×24 images

| kernel | padding | stride | output |
|---|---|---|---|
| 3 | 0 | 1 | 22 |
| 3 | 1 | 1 | **24** |
| 5 | 2 | 1 | **24** |
| 3 | 1 | 2 | **12** |

::: notes
Walk two of these out loud. Row one: 24 plus nothing, minus 3, over 1, plus 1
is 22 - the two lost rows are the positions where a 3×3 window would hang over
the edge. Row two: 24 plus 2, minus 3, plus 1 is 24, and the padding has bought
those two rows back.

Every one of these was measured in notebook 01 by building the layer and
printing the shape, not by trusting the formula. That is a habit worth copying:
the formula is easy to misremember by one, and the shape is free to print.

Ask the room for row four before revealing it. The common wrong answer is 11,
from forgetting the plus one; the second most common is 13, from adding one
before the floor. Handout section 3.2.
:::

# Two cases cover almost everything

- **3×3 with padding 1 leaves the size unchanged**: Keras calls this `padding="same"`
- In general, padding = (kernel − 1) / 2 for odd kernels
- **Stride 2 halves it**: the cheap alternative to pooling
- Almost every architecture you will read is built from these two

::: notes
These two rules are what you actually need at the keyboard. Same-padding
convolutions to compute, stride or pooling to shrink. If you internalise
nothing else from the formula, internalise that a 3×3 with padding 1 is
size-preserving, so you can stack as many as you like without arithmetic.

Say why odd kernels dominate in practice: the padding formula only gives a
whole number for odd f, and an odd kernel has a well-defined centre pixel, so
the output is aligned with the input rather than offset by half a pixel. That
is why you see 3×3, 5×5 and 7×7 everywhere and almost never 4×4.

Today's network uses same-padding 3×3 convolutions and 2×2 pooling, so the
shapes go 24, 24, 12, 12, 6. Handout section 3.2.
:::

# Four kernels written down, not learned

![](convolution_by_hand.png)

::: notes
Nothing here is trained. Four sets of nine numbers, chosen by hand, applied to
the same defective die. Left to right: a local average, which blurs and hides
the defect entirely; two edge detectors, which respond to the background
gradient as strongly as to the scratch; and a centre-surround contrast
detector, which isolates it.

Dwell on the failures rather than the success, because the failures are the
argument. The averaging kernel destroys exactly the information we need. The
edge detectors respond wherever brightness changes, and the film thickness
changes brightness everywhere, so they light up across the whole die.

Only the fourth panel has a clean bright spot on the defect and near-zero
elsewhere. Ask the room what is different about those nine numbers before the
next slide answers it. The answer is a property of their sum. Handout
section 3.3.
:::

# A constant patch gives exactly zero

$$\sum_{i,j} v \cdot K[i,j] = v \sum_{i,j} K[i,j] = v \cdot 0 = 0$$

::: notes
The fourth kernel is +8 in the centre and −1 in each of the eight surrounding
positions, so its nine weights sum to zero. Feed it any patch in which all nine
pixels have the same value v, and the response is v times zero, whatever v is.

This is the algebra behind the fourth panel. A zero-sum kernel is *blind to
absolute brightness* - it cannot see the film thickness, because a smoothly
varying background is locally almost constant. It responds only to how much a
pixel differs from its own neighbourhood, which is precisely the definition of
a defect on this line.

Nine numbers, chosen in advance, and the hardest part of the problem - the
varying background that defeats every brightness threshold - is simply gone.
Handout section 3.3.
:::

# Nine numbers, no training: 3.129 against 1.118

- Centre +8, each of the eight neighbours −1, sum exactly zero
- Blind to brightness, sensitive only to **local contrast**
- Strongest response on a defective die: **3.129**
- Strongest response on a clean die: **1.118**
- Nothing has been trained yet

::: notes
Say the last bullet twice. No gradient descent, no labels, no training set  - 
nine numbers written down from an argument about what the problem requires, and
the two classes are already separated by a factor of nearly three.

This is the honest baseline that the rest of the lesson has to beat, and it is
also a preview of the course's closing argument: whoever computes the right
feature has done most of the work. In section 12 we will hand four numbers
derived from this kernel to logistic regression, k-nearest neighbours and a
support vector machine, and all three will hit the ceiling.

One forward pointer to hold them: when notebook 02 trains a network and we look
at the eight kernels its first layer learned, seven of the eight will sum to
nearly zero. Nobody told it to. Handout sections 3.3 and 12.
:::

# Notebook 1, live

- The convolution in eleven lines, checked against scipy
- Output sizes measured, then compared with the formula
- Four kernels by hand, and the zero-sum one that works
- Equivariance checked at four offsets, then pooling

::: notes
Run `Notebooks/01_what_a_convolution_is.ipynb`. Twenty minutes.

The cell to protect if time runs short is the hand-written kernel comparison,
because watching the contrast kernel isolate a defect that the edge detectors
smear across the whole die is what makes the zero-sum argument concrete rather
than algebraic.

Second priority is the equivariance check, which prints a difference of exactly
zero rather than something small. Have them predict what it will print first  - 
most expect a tiny floating-point residual, and the fact that it is bit-exact
away from the borders says something stronger than "approximately equal".

Have them change the kernel: ask what happens to the response if they scale all
nine weights by ten, and whether that changes which dies are flagged.
:::

# Break

- Twelve minutes

::: notes
Twelve minutes. What comes back after the break is the property the whole idea
rests on, and then the experiment this lesson exists for - the one that
produces today's number. Worth saying so before they leave the room.
:::

# Shift the input, and the response shifts with it

- **Equivariance**: the response does not change, it **moves**
- It follows in one line from the definition
- Notebook 01 checks it at four offsets: largest difference exactly **0**
- A dense layer has nothing of the kind
- Shift by one pixel and all 576 weights meet different pixels

::: notes
Define equivariance carefully, because it is routinely confused with
invariance, and today's architecture uses both - one in the convolution and one
in the pooling. Equivariant means the output changes in a predictable,
corresponding way: shift the input by four pixels and the feature map shifts by
four pixels, unchanged in value.

The proof is one line from the definition: the sum at the shifted position runs
over exactly the pixels the unshifted sum ran over. But notebook 01 checks it
rather than believing it, and the answer is not "small" - away from the borders
it is bit-for-bit identical.

Then say what a dense unit does under the same shift. Its output is not shifted.
It is unrelated: a different set of pixels now multiplies each weight, and there
is no relationship between the two answers at all. Handout section 4.1.
:::

# Shift, then convolve, is convolve, then shift

$$(\text{shift}_{d}\,I) * K = \text{shift}_{d}(I * K)$$

::: notes
Read both sides in words. On the left: move the die, then run the detector. On
the right: run the detector, then move the answer. They are the same thing, and
that identity is the entire reason a convolutional network does not have to be
taught about positions.

Point out what the equation does *not* say. It does not say the answer is
unchanged by a shift - the feature map moves. Turning "it moved" into "it does
not matter where it was" is a separate operation, and it is the next segment:
pooling.

Worth one sentence on the borders, since the notebook shows it: at the edge of
the image the shift has nowhere to come from, so the identity holds in the
interior and not on the boundary. Every practical implementation makes some
choice there, and padding is that choice. Handout section 4.1.
:::

# The bright spot tracks the defect

![](equivariance.png)

::: notes
Top row: the same die, shifted by four different amounts. Bottom row: each
one's response to the contrast kernel written down before the break. The bright
spot in the response is directly under the defect in every panel.

The sentence to say slowly: no retraining occurs between these panels, because
there is nothing to retrain. The nine weights are identical across all four
columns. The detector was never told where to look, so it cannot be surprised
by where the defect is.

Ask the room to imagine the same figure for a dense unit. Panel one fires;
panels two, three and four are whatever those unrelated weights happen to
produce, which is noise. Then tell them we are about to measure exactly that,
and it is worse than noise - it is below chance. Handout section 4.1.
:::

# Pooling: from *where* to *whether*

- Equivariance says the response **moves** with the defect
- To grade a die we do not want it to move
- We want one number: did a strong response occur **anywhere**?
- Max pooling takes the maximum over a window, discarding the position
- 24×24 down to 3×3, and the maximum is still 3.129

::: notes
Frame pooling as answering a different question rather than as downsampling,
because "shrinking the image to save compute" is the common and much weaker
reading of it.

The grading task asks whether a defect is present, not where. Equivariance
gives us a feature map in which the answer has moved; taking a maximum over
that map throws the position away and keeps the strength. The question changes
from *where* to *whether*, and that is exactly the question the station asks.

The 3.129 is the same number from the hand-written kernel: pool that feature map
three times, from 24×24 to 12×12 to 6×6 to 3×3, and the maximum is 3.129 at
every stage. Of course it is - that is what taking a maximum does. Say it
anyway, because seeing it survive four sizes is what makes pooling feel safe.
Handout section 4.2.
:::

# Pooled three times; the maximum never moves

![](pooling.png)

::: notes
One feature map, pooled repeatedly. Read the maximum printed on each panel: it
is 3.129 in all four. The map shrinks by a factor of two each time and loses
its detail, and the one number we care about is untouched.

Point at what is lost as well as what is kept, because the loss is real. By the
3×3 panel you can no longer say where on the die the defect was, only that
something scored 3.129 somewhere in a quadrant. For a pass or fail grade that
is free. For a system that must tell a technician which corner to inspect, it
is a serious cost.

Ask the room which tasks in their own experience would be damaged by that.
Segmentation, object detection and keypoint location all need *where*, and all
of them use architectures that undo this pooling later. Handout section 4.2.
:::

# Three things pooling does

- Turns equivariance into **invariance**: one pixel of shift often changes nothing
- Enlarges the receptive field: 3×3 after pooling covers 6×6 of the image
- Discards spatial precision, which is a **cost** when *where* is the answer
- **Global** max pooling: one number per kernel, over the whole map

::: notes
Separate the three, because they are usually presented as one benefit and only
the first is the reason we are using it today.

Invariance is the goal here: after pooling, moving the defect by a pixel often
produces a literally identical output. Receptive field growth is the reason
depth works at all - stack pooling and small kernels and a 3×3 window at layer
four sees most of the die, for nine weights. Precision loss is the bill.

The last bullet is the design choice that makes today's headline result
possible. Global max pooling reduces each of the 16 feature maps to a single
number: did this kernel ever fire strongly, anywhere on this die? There is no
position left in the representation for the dense head to depend on, which is
why the network at 1:35 does not care where the defect is.

Handout section 4.2.
:::

# 1,537 parameters, in full

| layer | output shape | parameters |
|---|---|---|
| conv, 8 kernels of 3×3, same | 24 × 24 × 8 | 80 |
| max pool 2×2 | 12 × 12 × 8 | 0 |
| conv, 16 kernels of 3×3, same | 12 × 12 × 16 | 1,168 |
| max pool 2×2 | 6 × 6 × 16 | 0 |
| global max pool | 16 | 0 |
| dense 16, then dense 1 | 1 | 289 |
| **total** | | **1,537** |

::: notes
Read the shape column down the middle: 24, 12, 6, then 16 numbers, then one.
The image is progressively traded for depth, and then the global maximum
collapses the spatial dimensions entirely.

Against this, lesson 9's dense network on the same images has 213,761
parameters - a factor of 139. Both are trained with the same optimiser, the same
loss and the same epoch budget throughout this lesson, which is what makes
every comparison today about architecture rather than about tuning.

The two dense layers at the end are only 289 parameters because they sit
*after* the global maximum, on 16 numbers rather than on an image. That
ordering is the whole architecture: convolutions ask local questions, the
maximum discards position, and only then does anything dense happen. Handout
section 5.
:::

# The count students get wrong: 1,168

- Each of the 16 kernels is 3×3 **across all eight input channels**
- So each one holds 9 × 8 = 72 weights, not 9
- 16 × 72 + 16 = 1,168
- A kernel's depth always matches the input's channel count
- Only its two spatial dimensions are yours to choose

::: notes
This is the arithmetic worth stopping on, because nearly everybody computes
16 × 9 + 16 = 160 the first time and then cannot reconcile the framework's
summary with their own count.

The rule is the fourth bullet, and it is worth stating as a rule rather than as
an example: a convolutional kernel is always as deep as its input. On a colour
photograph the first layer's 3×3 kernels are 3×3×3. Here the second layer's
input has eight channels because the first layer had eight kernels, so its
kernels are 3×3×8. You choose the two spatial dimensions and the number of
kernels; the depth is chosen for you.

Ask the room what the third convolutional layer would cost if we added one with
32 kernels of 3×3. It is 32 × (9 × 16) + 32 = 4,640. Handout section 5.
:::

# Head to head, and where the missing 2% lives

- Dense network, lesson 9: **0.6928**, standard deviation 0.0342
- Convolutional: **0.9800**, standard deviation 0.0000
- Against the **true** grade the convolutional network scores **1.0000**
- Its visible 0.98 is the grading station's error and nothing else
- There is no gap left for a better architecture to close

::: notes
Same images, same optimiser, same epoch budget, 3,000 training and 2,000 test.
Twenty-nine points apart, with the smaller model on top by a factor of 139 in
parameters.

But the third bullet is the one that matters, and it is only available because
the data is synthetic and publishes its own truth. Scored against what the die
*actually* was, rather than against what the station recorded, the
convolutional network is right about every single die in the test set. Its 0.98
is not a modelling shortfall of any kind.

Say the consequence out loud, because it is a professional habit rather than a
fact about this dataset: any effort spent pushing this model from 0.98 to 0.99
would be effort spent learning to reproduce the station's mistakes. Knowing
where your ceiling comes from is what tells you when to stop. Handout
section 5.1.
:::

# 1,537 parameters against 213,761

![](dense_vs_conv.png)

::: notes
Left panel: test accuracy, 0.6928 against 0.9800. Right panel: parameter count,
on a logarithmic scale because otherwise the convolutional bar is invisible.

The pairing is the point. Every other lesson in this course has traded capacity
for performance in the expected direction - more trees, more units, more
flexibility. Here the smaller model wins by twenty-nine points, and it wins
because of what it cannot do rather than what it can.

Note the standard deviation of 0.0000 on the convolutional side across seeds.
That is not a rounding artefact and it is not luck: every seed reaches the
ceiling, because there is nothing difficult left in the problem once position
stops mattering. Compare it with the dense network's 0.0342, which is three and
a half points of seed-to-seed variation. Handout section 5.1.
:::

# A defect where none has been seen

- Training defects live only in the **top** band of rows
- One test set keeps them there; the other moves them to the **bottom**
- Same generator, same defect, same contrast, same number of images
- Only the row changed
- Predict the two numbers before you see them

::: notes
Set the experiment up properly and take predictions before revealing anything,
because the room's prediction is itself part of the lesson and section 6.1 of
the handout is about why the reasonable prediction is wrong.

Everything is held fixed except the band of rows the defect occupies. Same code
generates both test sets; the defect is identical in size, contrast and shape.
This is a controlled experiment about position and nothing else.

Take a show of hands on the dense network's score on the moved band. Most rooms
predict a drop of ten or twenty points - the reasoning being that it has
learned something about scratches in general and will retain part of it. That
reasoning is sound for almost every kind of distribution shift you will meet,
and it is wrong here. Handout section 6.
:::

# The number to carry out of the room

| network | trained band | moved to the other half | cost |
|---|---|---|---|
| dense (lesson 9) | 0.8567 | **0.4617** | **−0.3950** |
| convolutional | 0.9800 | 0.9847 | +0.0047 |

::: notes
Say it slowly and then say it again. A dense network scoring 0.8567 on defects
in the half of the die it was trained on scores 0.4617 - **below chance** - on
the identical defect moved to the other half. The convolutional network goes
from 0.9800 to 0.9847, which is noise around the ceiling.

Below chance deserves its own sentence. The dense network is not confused by
the moved defect; it is confidently wrong about it, because the pixels that
used to carry evidence of "clean" are now the ones carrying the defect. Nothing
degraded gracefully. Thirty-nine and a half points, from moving a scratch.

This is the sixth carry-home number of the course and the last one it will
produce. Ask them to write it down now: 0.8567 to 0.4617, against 0.9800 to
0.9847. Everything else today is commentary on that line. Handout section 6.
:::

# Where the defects were, and what it cost

![](translation_test.png)

::: notes
Left and centre panels: forty defective dies from each band, averaged, so the
bright smear shows exactly where the defects lived. Top band for training,
bottom band for the moved test set. There is no overlap at all, which is what
makes this a clean test rather than a partial one.

Right panel: the result from the previous slide, drawn. The dense network's bar
falls through the chance line; the convolutional network's does not move.

Point at the averaged panels once more and note what they say about the dense
network's task. During training, every defect it ever saw was in the top third
of the image. It had no reason whatsoever to build a detector anywhere else,
and no mechanism that would have shared one if it had. Handout section 6.
:::

# Below chance, and why nothing degraded

- Most students predict a drop that stops well above chance
- The reasoning is sound: shifts usually degrade, they do not vanish
- But this is not a change of **degree** in the inputs
- It is a change of **which** inputs carry the signal
- Knowledge stored per position, at positions never trained

::: notes
This is the predictable mistake of the lesson, and the students who made it
should hear why their instinct was reasonable rather than merely be corrected.

Almost every distribution shift you will meet is a change of degree: sensors
drift, populations age, prices inflate. A model trained on the old regime
degrades on the new one and usually retains something, because the features
still mean roughly what they meant.

Here the features do not mean anything different - they are simply not the
features carrying the signal any more. Pixel 100 was the evidence; pixel 400 is
now the evidence; and pixel 400's weights were trained exclusively on clean
silicon. There is no partial credit available.

Ask them where else this failure shape occurs. A time-series model given
reordered columns, a tabular model after a schema change: same category of
break. Handout section 6.1.
:::

# A convolutional layer is a restricted dense layer

- Every function it computes, a dense layer can compute too
- Tie the shared entries, zero the rest: it is strictly **less** expressive
- The functions it gives up all treat row 5 differently from row 15
- Nobody wanted any of those
- **A model that cannot express a wrong answer need not learn to avoid it**

::: notes
This is the intellectual core of the lesson and the argument runs opposite to
the intuitive direction, so give it time.

The convolution is not a more powerful layer. It is a dense layer with most of
its weights set to zero and the rest tied together in a particular pattern.
Strictly fewer functions. On any expressiveness measure it loses.

And it wins by thirty-nine points, because the functions it surrendered were
precisely the position-dependent ones, and every one of those was wrong on this
problem. The dense network had to *learn* not to depend on position, from data
that never showed it a defect low on the die, and it failed.

Put the general question to the room, because it is the question the whole
course has been circling: what assumption does your model encode, and is it
true? Handout section 6.
:::

# Augmentation buys 8 points and leaves it 44 short

- Show the dense network shifted copies: a moved scratch is still a scratch
- Dense: 0.4617 → **0.5423**, a real gain of **+0.0807**
- Convolutional: 0.9847 → 0.9847, unchanged, with nothing to gain
- Augmentation **teaches** an invariance: approximate, and forgettable
- Architecture **asserts** it: exact, free, permanent

::: notes
The obvious repair for the previous result is to manufacture the missing
examples: take the training images and shift them, so the network sees defects
lower down. It works, and it is worth eight points, and it is still forty-four
points behind what the architecture provides for nothing.

Two details of the experiment are worth copying, because both are easy to get
wrong in a way that flatters your conclusion. Both arms got the same number of
epochs - an augmented set is five times larger, so equal wall-clock would have
quietly given it five times fewer passes per image; an earlier version of this
experiment did that and reported 1.6 points instead of 8.1. And the shift is
drawn per image, not per batch: one offset applied to a whole copy just adds
four more fixed positions.

Then give the rule: augment for invariances no layer can express - brightness,
contrast, small rotations - where it is indispensable. Handout section 7.
:::

# At the ceiling by 500 images; 0.235 short at 8,000

![](sample_efficiency.png)

::: notes
Test accuracy against training-set size, logarithmic horizontal axis. The
convolutional network reaches the ceiling at 500 images and stays flat. The
dense network at 8,000 - sixteen times more data - is still 0.235 below it.

Warn them about the flat line before someone asks, because it looks like a
plotting bug. It is not: the convolutional network is already right about every
die at 500 images, and the only thing between it and 1.0 is the station's 2%.
There is nowhere left for the curve to go.

This is the price of the weight-sharing assumption, paid in the only currency
that matters. The dense network is not incapable of this problem; it is
required to learn separately, at every position, what the convolution learned
once. Ask the room what they would rather have on a new project: a better
optimiser, or a correct assumption. Handout section 8.
:::

# Notebook 2, live

- Both architectures trained head to head, with seed spreads
- The translation experiment, run in front of you
- Augmentation with the epoch budget equalised
- The learning curve from 100 images to 8,000
- The permutation test: hold your prediction

::: notes
Run `Notebooks/02_convnets_on_wafers.ipynb`. Twenty-two minutes.

The cell to protect is the translation experiment, since it produces today's
number. Have them run it with their own seed and confirm that the dense
network lands below 0.5 rather than near it.

Second priority is the augmentation comparison, specifically the epoch
argument. Show them the two lines of code that equalise the passes per image
and ask what the result would have been without them - it is in the notebook,
and it is 1.6 points instead of 8.1. That is a lesson about experimental
hygiene rather than about augmentation, and it is the lesson 5 habit applied to
deep learning.

The permutation cells come last. Ask for predictions before running them: the
room will be wrong, and that is the point of the final segment.
:::

# A new defect appears on a Monday

- A contamination **cluster**: several faint specks together
- By Friday, a few dozen labelled photographs
- Section 8 says the network needs several hundred
- Waiting six months is no answer: the line ships now
- Can we borrow from a network trained on something else?

::: notes
Set up transfer learning as the response to a scheduling problem rather than as
a technique in its own right, because that is what it is. The physics of the
line changed on Monday; the label budget has not caught up; the sample
efficiency curve you saw twenty minutes ago says the model needs data nobody
has yet.

The new defect is genuinely different from the old ones: faint, bright rather
than dark, and spread over several disconnected specks rather than a single
line. Say that now, because it is what makes the negative-transfer result at
the end of the segment work.

Ask the room what they would try first. Most say "fine-tune a pre-trained
model", which is right, and then the interesting questions are the two this
segment measures: pre-trained on what, and with which layers frozen. Handout
section 10.
:::

# The defect the line knows, and Monday's

![](rare_defect.png)

::: notes
Left: the scratch, dark, linear, seven pixels, the thing every model in this
lesson has been trained on. Right: the cluster, a handful of faint bright
specks with no particular shape.

Point at the polarity difference explicitly and make sure it registers, because
it is the mechanism behind the last result of this segment. The known defect is
*darker* than its surroundings. The new one is *brighter*. Any detector that
has committed to reporting local darkness is not merely uninformed about the
new defect - it is pointed the wrong way.

Ask the room whether a network trained to grade scratches should transfer well
here. The honest answer is "it depends entirely on what the source task forced
it to learn", and that sentence is the whole content of the next slide.
Handout section 10.
:::

# Start from someone else's weights

- **Transfer learning**: initialise from a network trained on a related task
- The argument: early layers detect local contrast, which is not scratch-specific
- Two knobs: which **source** task, and which layers stay **frozen**
- Both knobs matter more than the received wisdom suggests

::: notes
Define it plainly: instead of starting from random numbers, start from the
weights of a network that already learned something on a related problem, then
continue training on your small dataset.

The justification is architectural rather than mystical. The first layer of any
image network learns local contrast operators - we saw eight of them
rediscovering the zero-sum trick - and there is nothing about local contrast
that is specific to scratches. If that is true, those layers are worth
borrowing.

Flag the two knobs now, because the next three slides are one measurement each
of them. Which source task, and how much of the borrowed network you allow to
change. The received wisdom on both - "any related task", "freeze the base"  - 
is what this segment tests. Handout section 10.
:::

# The source task is the whole game

- Pre-train on a task **nobody at Meridian wants solved**
- Naming which of three types a die carries: clean, scratch, particle
- Grading pass/fail teaches **one** detector; typing forces description
- **0.9993** on the source task: a receipt, not a result
- Transfer carries what the source **forced** it to learn

::: notes
This is the design decision that makes the rest of the segment work, and it is
counter-intuitive enough to state twice: we deliberately pre-train on a task
that has no business value, because it demands richer features than the task we
actually care about.

Pass/fail grading on scratches can be solved with a single dark-line detector.
Naming which of three types is present forces the early layers to describe dark
lines *and* bright blobs, and to tell them apart. Those are the features worth
carrying to a new bright defect.

The 0.9993 is not a boast. It is a receipt: it confirms the source task was
actually learned, which is the precondition for its features being worth
anything at all. A pre-trained network that never solved its own task has
nothing to lend. Handout section 10.1.
:::

# A window, not a slope

| labelled images | from scratch | warm-started | gain |
|---|---|---|---|
| 25 | 0.5100 | 0.5100 | 0.0000 |
| 100 | 0.7220 | **0.8640** | **+0.1420** |
| 200 | 0.7270 | **0.8940** | **+0.1670** |
| 400 | 0.8447 | 0.7877 | **−0.0570** |

::: notes
The received wisdom is that transfer helps most when data is scarcest. These
numbers do not say that, and the shape they do describe is more useful.

At 25 images nothing works - both arms sit at chance. There is too little data
to fit even a small head onto good features, so the borrowed weights have
nothing to attach to. Between 50 and 200 the warm start is worth 9 to 17 points
and the gain *grows* across that range. By 400 it is negative: the from-scratch
network has now seen enough of the new defect to learn its own features, and
the borrowed ones are a constraint rather than a head start.

So transfer occupies a window. Too little data and it has nothing to attach to,
too much and it has nothing to add. Ask the room how they would find that
window on a real problem where they cannot see this table. The answer is the
lesson 5 answer: measure it. Handout section 10.2.
:::

# The window, drawn

![](transfer_curve.png)

::: notes
The same table as a picture. Two things to point at.

First, the gap between the warm-started curve and the from-scratch curve opens
and then closes. That is the window. If you had run this experiment at a single
dataset size - which is what most people do - you could have concluded anything
you liked, including that transfer is useless, depending on which size you
picked.

Second, the frozen-base curve, which is below the from-scratch curve at 50 and
100 images. That is the surprise of the next slide, and it is worth letting
them see it here first and ask about it themselves.

The methodological point is one lesson 5 made: a single operating point is not
a finding. The shape of the curve was the finding, and it needed five sizes to
appear. Handout section 10.2.
:::

# When transfer makes things worse

| labelled images | from scratch | frozen, typing source | frozen, scratches only |
|---|---|---|---|
| 50 | 0.5707 | 0.5273 | 0.5150 |
| 100 | **0.7220** | 0.5390 | 0.5047 |
| 200 | 0.7270 | 0.7150 | 0.5350 |

::: notes
Read the right-hand column first. A base pre-trained only on scratches, then
frozen and pointed at the bright cluster defect, never leaves chance at any
dataset size. This is **negative transfer**, and it is real rather than
theoretical: those kernels are committed to reporting local darkness where the
new defect is bright, and a frozen layer cannot change its mind.

Now read the middle column, which was included as a control and turned out to
be the more interesting result. The *good* source, the typing task, is also
behind starting from noise at 50 and 100 images when frozen, and only pulls
level at 200. Yet the same source, unfrozen, was fourteen points ahead at
exactly 100 images.

So the features were useful. Freezing them was the mistake. Ask the room which
of the two columns they would have predicted, and how many of them would have
frozen the base by default. Handout section 10.3.
:::

# Two rules, and neither is "always pre-train"
- Transfer carries what the source task **forced** the network to learn
- Ask what the source required before trusting its features
- Freezing is a **bet** that the borrowed features are already right
- **In doubt: warm start, and leave everything trainable**

::: notes
Land the segment on two rules that are usable at the keyboard next week.

The first is about choosing a source. "Related task" is too weak a criterion.
The right question is what the source task made impossible to ignore. Our
typing task made it impossible to ignore bright blobs, which is why it
transferred to a bright defect; the scratch task did not, and did not.

The second is about freezing. Freezing asserts that the borrowed features need
no adjustment. When that is true it is cheap and it regularises. When it is
false there is no recovery, because gradients never reach the frozen layers. An
unfrozen warm start can recover from a mediocre source; a frozen base cannot.

The asymmetry is the argument: one option has a bad worst case and the other
does not. Handout section 10.3.
:::

# Notebook 3, live

- Pre-training on the task nobody wants solved, up to 0.9993
- The transfer window, measured at five dataset sizes
- Negative transfer, from a source chosen badly on purpose
- Every family the course has met, on one problem

::: notes
Run `Notebooks/03_transfer_and_synthesis.ipynb`. Eighteen minutes.

Two cells to protect. The five-size transfer sweep, because the window only
exists as a shape and any single row of it supports the wrong conclusion - and
because they can see for themselves how tempting it would have been to run one
size and publish.

The second is the final table, which is the course synthesis and the last cell
of the last notebook. Let it run and let them read it before you comment. The
random forest row is the one that changes the conversation, and it lands
harder if they find it themselves rather than being told.

If time is short, cut the negative transfer cell and keep the synthesis: the
numbers are already on the slide, and the synthesis is what the final segment
needs on screen.
:::

# What is the convolution actually using?
- "Images have spatial structure" is **two** claims, not one
- Permute every image's pixels identically, then train
- Detection: **0.9798 → 0.9800**, no cost at all
- Typing: **0.9970 → 0.9523**, 4.5 points
- The dense network notices neither

::: notes
This is the experiment lesson 9 set up in its closing sentence, and the result
is the one that surprised the person who wrote it.

Apply one fixed permutation to the 576 pixel positions, apply it to every image
in the dataset, and retrain from scratch. Adjacency is destroyed - neighbouring
pixels are now scattered across the image - while every pixel value is
preserved.

On grading a die, the convolution does not notice. Read that again: the
architecture defined entirely by adjacency loses nothing when adjacency is
destroyed. It looks wrong until you ask what the task requires - "is there a
patch of unusual local contrast somewhere" - and notice that a scattered
permutation leaves those pixels somewhere, still extreme. On naming the defect
type, which is a question about *shape*, the permutation costs 4.5 points,
because shape is exactly what it destroys. Handout section 9.
:::

# Nothing on detection, 4.5 points on typing

![](permutation_test.png)

::: notes
Four pairs of bars: two tasks, two architectures, as photographed against
permuted. Only one pair moves appreciably, and it is the convolutional network
on the typing task.

Walk through the reading once more with the picture in front of them, because
this is the most counter-intuitive result of the lesson. The dense pairs are
flat because a permutation is, for a dense layer, nothing but a relabelling of
input units. The convolutional detection pair is flat because the task never
needed adjacency. The convolutional typing pair drops because telling a line
from a blob is a question about arrangement, and the arrangement is gone.

Ask the room what this implies about a convolutional network's advantage on a
task they cannot inspect. It implies that "it uses spatial structure" is not an
explanation until you say which part of the structure, and for what. Handout
section 9.
:::

# Two assumptions, and only one is about arrangement

| assumption | what it says | needed by |
|---|---|---|
| **weight sharing** | a pattern means the same thing wherever it appears | both tasks |
| **locality** | nearby pixels belong together | only the typing task |

::: notes
This is the table to photograph. The usual one-line justification for
convolution bundles two assumptions, and this data separates them: on detection
the entire advantage came from weight sharing, and locality contributed
nothing measurable.

Say why that matters beyond this lesson. Weight sharing is not unique to
convolution - other architectures supply it in other forms, for sequences and
for sets. If your task needs only weight sharing, convolution is one option
among several, and the choice should be made on other grounds. If it needs
locality too, you are asking for something more specific.

Put the question to the room in its practical form: the next time you reach for
a convolution, which of these two are you buying? Anyone who cannot answer has
chosen a default rather than a model. Handout section 9.
:::

# What the first layer learned, unprompted

![](learned_kernels.png)

::: notes
The eight 3×3 kernels the first layer learned, red positive and blue negative.
Seven of the eight sum to less than 0.6 in absolute value, against a scale of
2.16 if all nine weights had pointed the same way.

Say what that means and let it close the loop with the first hour. Before the
break we wrote a zero-sum kernel by hand, from an argument about film thickness
and local contrast. Nothing in the loss function, the initialisation or the
architecture mentions that argument. Gradient descent rediscovered it, because
the data rewarded it.

This is the most honest thing that can be said in favour of learned
representations, and it cuts both ways. It found a good feature without being
told. It would equally have found a bad one had the data rewarded that - which
is every leakage story from lesson 1, in a new architecture. Handout sections
3.3 and 9.
:::

# What this lesson did not need
- Today ran on a laptop processor, no graphics card
- Deeper networks revive lesson 9's gradient problem
- **Batch normalisation** rescales each layer's activations
- **Residual connections** let the gradient skip a layer
- Neither is needed at 24×24

::: notes
Name what was skipped rather than pretending the field ends at two
convolutional layers, and name why it was skipped: a technique introduced
without a problem it solves is a technique nobody remembers.

Deep stacks attenuate gradients exactly as lesson 9's sigmoid layers did, and
the two standard repairs are on the slide. Batch normalisation keeps each
layer's activations at a sane scale so the signal neither dies nor explodes.
Residual connections add a layer's input to its output, so there is always a
path along which the gradient reaches earlier layers undiminished.

Both are direct descendants of what was derived last week, which is the point
worth making. Our images are 576 pixels with one defect, and 1,537 parameters
solve them perfectly, so neither repair would have anything to do here.
`Resources/` carries the pointers. Handout section 11.
:::

# Ten lessons on one problem

| family | raw pixels | 4 hand-made features |
|---|---|---|
| logistic regression (3, 4) | 0.8905 | 0.9800 |
| k-nearest neighbours (6) | **0.4970** | 0.9800 |
| support vector machine (6) | 0.9660 | 0.9800 |
| random forest (7) | **0.9780** | 0.9795 |
| dense network (9) | 0.7430 | - |
| convolutional network (10) | **0.9800** | - |

::: notes
This table was built expecting to show that classical methods cannot handle
images. It shows no such thing, and the honest reading is far more useful.

A random forest on raw pixels reaches 0.9780, within two thousandths of the
convolutional network, knowing nothing about which pixel adjoins which. It does
not need to: a defect drives *some* pixel to an unusual value, and 300 trees
splitting on individual pixels cover enough of them. That is the forest's own
inductive bias - the answer depends on a few features crossing thresholds - and
it happens to fit.

Two failures are instructive. k-nearest neighbours collapses to 0.4970 because
Euclidean distance over 576 pixels is dominated by the smooth background - the
curse of dimensionality from lesson 6, exactly. And lesson 9's dense network,
at 0.7430, sits *behind* plain logistic regression. Handout section 12.
:::

# Four hand-written numbers reach the ceiling

![](family_comparison.png)

::: notes
The same table, drawn, with the ceiling of 0.9800 marked. The right-hand
cluster is what to look at: four hand-made features computed with the contrast
kernel from the first hour, and every classical family lands at the ceiling.

Say what that costs and what it buys. It cost nine numbers and an argument
about film thickness - twenty minutes of thinking by someone who understood the
physics. It bought the entire problem, for every model family, including the
one that scored 0.4970 on raw pixels.

The convolutional network, given only raw pixels, found an equivalent feature
by itself, and paid for it in data and compute. Both routes end at the same
ceiling. Ask the room which one they would take on a problem where they
understood the physics, and which on a problem where they did not. That is the
real choice, and it is the last one this course asks them to make. Handout
section 12.
:::

# The representation matters more than the model

- A model turns a **representation** into a decision
- Classical methods make you build it; networks learn it, paid for in data
- Lesson 9: a hidden layer worth 39 points on one problem, 0.09 on another
- Today: a random forest within **0.002** of the convolutional network
- **Neither route is a default**

::: notes
This is the sentence the ten weeks add up to, and it deserves to be said slowly
and then written on the board.

Every lesson in this course has been an instance of it. Lesson 2's imputation
and scaling decisions changed more than any model choice that followed. Lesson
3's polynomial features turned a straight line into a curve without changing the
estimator at all. Lesson 6's kernel is a representation chosen in advance;
lesson 7's trees build one by splitting; lesson 8's components and embeddings
are representations with no labels at all.

The handout's closing table gives seven if-then rules for choosing - read it.
The one that matters most is the last: if you cannot say what assumption your
model encodes, you have not chosen a model, you have chosen a default.

Handout section 12 and 12.1.
:::

# Homework: the last one

- **Exercise 10**, due **Friday 4 December 2026, 23:59**
- `Exercises/10_convolutional_networks.md`
- It falls in the week of the **project peer review**: plan for both
- Two reviews to write as well as two to receive

::: notes
Set it explicitly and say the deadline out loud: Friday 4 December, 23:59. It
is the last exercise of the course.

Say the scheduling problem plainly rather than letting them discover it. That
same week is the project peer review week - each of them reads two other
projects and is read by two - and the reviews they write are assessed alongside
the project they submit. Two pieces of work in one week. Anyone who starts the
exercise on Thursday will do both badly.

Two standing requirements that today's material makes easy to forget. Any claim
that one architecture beats another needs the spread it was measured against.
And any accuracy needs the ceiling it was measured against - on this data that
is 0.98, and a model reported at 0.98 may be perfect rather than merely good.
:::

# What ten weeks add up to

- **0.8567 to 0.4617** for a dense network, on a defect moved half a die
- The convolutional one went 0.9800 to 0.9847, because it cannot ask *where*
- An assumption that is true beats flexibility that is not
- The representation is doing most of the work; the only question is who builds it
- Thank you, and good luck with the project

::: notes
Close on the number. A dense network scoring 0.8567 on defects in the half of
the die it was trained on scores 0.4617, below chance, on the identical defect
moved to the other half, while the convolutional network goes 0.9800 to 0.9847.
That is the sixth and last carry-home number of this course, and it is the
whole argument in one line: the convolutional network won by being *less*
expressive.

Then step back over the ten weeks. They arrived able to fit models. What this
course tried to add is the habit of asking what a number is measured against  - 
a ceiling, a spread, a baseline, a fold that was kept honest - and the habit of
naming the assumption a model encodes before trusting what it produces. Those
two habits are what separate a result from a plausible-looking result, and
nothing in this field has ever made them less necessary.

One forward-looking sentence, and only one: every system built on top of these
fundamentals - including the ones this degree programme covers elsewhere - is
still an argument about which inductive bias to encode and what to measure it
against, so the questions you have practised for ten weeks are the ones that
transfer.

Thank them, remind them of the project deadline and the office hours, and stop.
:::
