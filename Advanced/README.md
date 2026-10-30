# Advanced material

Lessons kept from earlier versions of this course, at a level beyond what a first
course in artificial intelligence asks for. They are **not taught** in this course and
**not examinable**: nothing in the exam draws on them, and the ten exercises do not
depend on them.

They are here for the curious, and for a later, more advanced course.

| Folder | What it adds to the lesson taught in its place |
|---|---|
| `08_unsupervised_learning_advanced/` | Lesson 8 as first written: k-means' convergence proved and k-means++ measured against random starts, four linkage rules compared on the bot sessions, whether DBSCAN's `eps` transfers to next week's data, principal component analysis derived from the variance-maximisation problem and computed a second way from the singular value decomposition. |
| `09_neural_networks_advanced/` | Lesson 9 as first written: backpropagation derived in matrix form and coded from scratch in NumPy, gradient checking, the variance argument behind weight initialisation, the vanishing gradient measured layer by layer, optimisers compared, and a yardstick of polygons for how many hidden units a region needs. |

Each folder keeps the layout of a lesson in `Lessons/` - handout, slides, notebooks,
quiz, exercise - and its notebooks run in the course container as they are. The course
tools (`tools/build.py`, `tools/verify_lesson.py`) work on `Lessons/` only, so this
material is not rebuilt or re-verified when the course changes. Notebook 01 of the
advanced lesson 9 takes several hours to run end to end.
