# Facial Recognition and Its Regulation

> **Supplementary reading — Lesson 10**
> Estimated reading time: 25 minutes
> Not examinable. The distinction in Section 1 — between checking a claim and
> searching a database — is arithmetic you already have from Lesson 4, and it is
> the single thing most often got wrong by people who buy these systems.

---

## Why a computer scientist should read this

Section 6 of the handout ran one experiment. Defects were confined to the top
band of the die during training; the test moved the identical defect to the
bottom band. The dense network went from **0.8567 to 0.4617** — below chance —
because every weight carrying the evidence was attached to a specific pixel and
those pixels were now different pixels.

The lesson drew the architectural conclusion. There is a second one, about
deployment rather than design: **a model's accuracy is a statement about the
distribution it was measured on, and nothing else.** On wafers, moving the input
somewhere the training set never went costs a batch of silicon. The same failure,
running on photographs of people, has put men in jail.

Almost every deployed face recognition system is Section 10 of the handout — a
convolutional network pre-trained on a very large collection of faces, with a
small head fitted for the task at hand. Everything in this course applies to it
directly. So does everything that goes wrong with it.

---

## 1. Two problems that share a name

Two things are called face recognition and they are not the same problem.

**Verification** is one-to-one. A phone or a passport gate holds one stored
template, you present a face, and the system answers a yes/no question about a
claim you made. The errors are the **false match rate (FMR)**, accepting an
impostor, and the **false non-match rate (FNMR)**, rejecting the right person.

**Identification** is one-to-many. A photograph is compared against a gallery —
a watchlist, a licence database, a mugshot archive — and the system returns a
ranked list of candidates. Nobody made a claim. The errors are the **false
positive identification rate (FPIR)** and the **false negative identification
rate (FNIR)**.

Why that decides everything is Lesson 4's material with different labels. In
verification you make **one** comparison per transaction; in identification you
make **N**, so a false match rate that is negligible per comparison is multiplied
by the gallery size before anyone sees a result.

Work it through — an illustration with stated assumptions, not a measurement of a
real system. Take a false match rate of **one in a million** per comparison,
which in verification would be called essentially never wrong, and a modest
gallery of **100,000** faces. The expected number of false candidates per search
is $100{,}000 \times 10^{-6} = 0.1$: roughly one search in ten returns a
stranger, an FPIR of about 9.5%.

Now add the base rate. Suppose that in one search out of a thousand the person
photographed is genuinely enrolled, and that when they are, the system finds them
95% of the time. Per thousand searches that is about 0.95 genuine hits against
about 95 false ones. **Roughly one candidate in a hundred is the right person.**
The algorithm has not changed; the base rate did the damage, exactly as it did to
the disk-drive classifier in Lesson 4.

Hold that ratio for Section 2, because the differentials the audits found are
differentials in *false positives*, and a false positive is the error that
produces a knock on the door.

---

## 2. What the audits measured, and how the picture changed

Two documents are cited constantly, usually together, and they measured different
things.

**Gender Shades** (Buolamwini and Gebru, 2018) tested three commercial **gender
classification** products — software that looks at a face and outputs a category —
against a benchmark of parliamentarians chosen to span skin tones. The headline
was intersectional: error rates up to **34.7%** for darker-skinned women against a
maximum of **0.8%** for lighter-skinned men, in the same product at the same
time. A follow-up a year later found the audited vendors had substantially closed
those gaps — itself a finding: public measurement moved the products.

**Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects** (NISTIR 8280,
December 2019) is the United States National Institute of Standards and
Technology's (NIST) study of **recognition**, not classification. It ran
**18.27 million images of 8.49 million people** through **189 algorithms from 99
developers** on operational government photographs. Its central finding: false
positive differentials are far larger than false negative ones and are
widespread — "across demographics, false positive rates often vary by factors of
10 to beyond 100 times." On high-quality application photographs they were
highest for West African, East African and East Asian faces and lowest for
Eastern European faces; higher for women than men, consistently, though by less
than the effect of race; and higher for the very old and the very young.

Three details get dropped in the retelling.

**NIST explicitly separated the two literatures.** Its background section notes
that much public discussion cites the gender-classification studies, and that
"those studies did not evaluate face recognition algorithms, yet the results have
been widely cited to indict their accuracy." Quoting 34.7% as a face-recognition
error rate is simply wrong; the recognition differentials are real and were
measured separately.

**The effect was not universal across vendors.** Several algorithms developed in
China showed the *reverse* pattern, with low false positive rates on East Asian
faces — which tells you the differential tracks the training data rather than
anything about faces. NIST also recorded developers whose identification
algorithms were accurate enough that the differentials were undetectable.

**The trend moved.** NIST reported identification accuracy improving by roughly a
factor of twenty between 2014 and 2018, and its 2022 follow-up (NISTIR 8429)
states the relationship plainly: higher overall accuracy generally means smaller
differences across demographic groups. Quoting a 2019 differential as the state
of the art in 2026 is the same error as quoting a benchmark for a model you have
not retrained. The honest statement is not "face recognition is 34% wrong on
Black women". It is that accuracy varies enormously between vendors and
thresholds, that the weak systems fail unequally, and that **the system you are
deploying must be measured on the population you are deploying it on** — NIST's
own recommendation to system owners.

---

## 3. Where the faces came from

Section 10 of the handout showed a warm start worth up to 17 points, and made the
point that transfer carries whatever the source task forced the network to learn.
In face recognition the source task is identity discrimination over an enormous
gallery, and the source data is photographs of millions of real people. Whose?

For most of the last decade: photographs taken from the internet without asking.
**MegaFace**, assembled at the University of Washington in 2015, drew millions of
faces from a Flickr collection of Creative Commons-licensed images; independent
analysis found a large majority of those images carried licences prohibiting
commercial use, while the dataset was in fact used by major commercial
laboratories. It was withdrawn in 2020. **MS-Celeb-1M**, released by Microsoft
Research in 2016 with roughly ten million images of about a hundred thousand
"celebrities", was withdrawn in 2019 after reporting showed many of them were
ordinary people who happened to appear in the news. Neither withdrawal recalls
the models already trained on them.

The legal shape of this is not the one students expect. A Creative Commons licence
grants permission to *copy the photograph*. It says nothing about the depicted
person, who was not a party to it and whose face is personal data about them.
**Copyright clearance and data protection clearance are different clearances, and
having one is no evidence of the other.**

The commercial version is **Clearview AI**, which built a searchable index of
billions of face images scraped from the public web and sold access to police
forces. European regulators reached the same conclusion independently and
repeatedly: the Italian, French and Greek authorities each fined it €20 million
in 2022, the Dutch authority €30.5 million in 2024. In the United States, with no
equivalent national statute, the constraint came from one state — Illinois's
Biometric Information Privacy Act, which requires informed written consent before
capturing biometric identifiers and, unusually, lets individuals sue. A 2022
settlement under it barred Clearview from selling to most private parties
nationwide; the same statute produced a $650 million settlement against Facebook
in 2021, after which the company shut down face tagging and deleted over a
billion stored templates.

---

## 4. The ground truth is a person's opinion

Section 5.1 of the handout drew a distinction the course returns to constantly:
the convolutional network scored 0.9800 against the **recorded** grade and 1.0000
against the **true** grade, and the gap was the grading station's 2% error, not a
modelling shortfall. Agreement with a fallible human grader is not agreement with
the truth.

In face identification there is no true grade at all. The system returns a ranked
list; a human examiner decides whether the top candidate is the same person; that
decision becomes the identification of record. The label is somebody's judgement,
formed while looking at the machine's suggestion — the worst possible order,
because it invites the examiner to confirm rather than to decide.

The consequences are documented. The American Civil Liberties Union maintains a
list of people in the United States wrongfully arrested after a face recognition
search; as of 2026 it names fourteen, and most of them are Black. The first
publicly known is Robert Williams, arrested in Detroit in January 2020 in front
of his family for a shoplifting he had nothing to do with. Others: Michael
Oliver, whose arms were covered in tattoos the suspect did not have; Randal Reid,
arrested in Georgia in 2022 on a Louisiana warrant for a place he had never been;
Porcha Woodruff, arrested in Detroit in 2023 while eight months pregnant; and
Christopher Gatlin, who spent over a year in jail before the case was dropped.

Two mechanisms recur. **The corroboration was not independent**: in at least half
these cases the confirming evidence was a photo lineup shown to a witness, built
around the photograph the algorithm had already selected. That is leakage —
Lesson 1's failure, in a police station. And **the match was treated as an
identification rather than a lead**: a *Washington Post* investigation published
in 2025 reviewed records from 23 departments that keep them and found 15, across
12 states, that had arrested someone on a face recognition match with no
independent evidence connecting them to the crime. Every vendor's documentation
says the output is an investigative lead. The documentation is not the
deployment.

The remedy actually imposed is procedural, not technical. Under the settlement of
Robert Williams's lawsuit, Detroit police may no longer seek an arrest warrant on
the combination of a face recognition lead and a photo lineup built from it. That
is a rule about the *pipeline* rather than the model — and the kind of rule an
engineer is far better placed to write than a lawyer.

---

## 5. What the law does

Four regimes are worth knowing, and they diverge sharply.

**The General Data Protection Regulation (GDPR)** treats biometric data processed
*for the purpose of uniquely identifying a natural person* as a special category
under Article 9, which begins from a prohibition and then lists narrow grounds
that lift it — explicit consent, or a substantial public interest with a basis in
Union or Member State law, among others. The qualifier is the part engineers
miss: the Regulation's recitals make clear that photographs are not
systematically special-category data. **The face becomes biometric data at the
moment you run the extractor over it**, so storing holiday photos and running a
recognition model over the same photos are legally different acts. Policing sits
outside the GDPR in a parallel instrument, the Law Enforcement Directive, which
is why the police cases are argued differently from the commercial ones.

**The EU Artificial Intelligence (AI) Act** (Regulation (EU) 2024/1689) adds
technology-specific rules on top. Its Article 5, the list of outright prohibited
practices, has applied since February 2025. It bans building or expanding facial
recognition databases through the untargeted scraping of face images from the
internet or closed-circuit television (CCTV) footage — the Clearview business
model, named and forbidden. It bans biometric categorisation inferring race,
political opinions, trade union membership, religious belief or sexual
orientation. And it prohibits **real-time** remote biometric identification in
publicly accessible spaces for law enforcement, subject to three narrow
exceptions — searching for specific victims of abduction, trafficking or
disappearance; preventing an imminent threat to life or a foreseeable terrorist
attack; and locating a suspect in an enumerated serious offence — each needing
prior authorisation from a judicial or independent administrative authority.

Note what is *not* prohibited: retrospective identification from recorded
footage, which is how most of the wrongful arrests above happened. That is
high-risk rather than banned, and the obligations attached to it were pushed back
to December 2027 by an amending instrument agreed in 2026.

**The United Kingdom** has no equivalent statute and got its rules from a court.
In *R (Bridges) v Chief Constable of South Wales Police* (2020) the Court of
Appeal held a live deployment unlawful on three grounds: the legal framework left
too much discretion to individual officers over who went on a watchlist and where
cameras were deployed; the data protection impact assessment was inadequate; and
the force had not discharged its public sector equality duty, having never
satisfied itself the software did not have a racial or sex bias. That third
ground is the direct legal descendant of Section 2 — an audit result, converted
into a duty.

**The United States** has no federal rule and diverges by city. San Francisco
banned government use in 2019; Portland, Oregon banned private use in places of
public accommodation too. The pattern since is not one-directional: New Orleans
repealed its 2020 ban in 2022 for violent-crime investigations, and reporting in
2024 found police in banned jurisdictions asking neighbouring agencies to run
searches for them — a ban on an office, not on a capability. By 2026 roughly
fifteen states restrict police use and a similar number of cities have bans.

The through-line: **the jurisdiction decides whether your system is a product or
a crime, and the same code is both.**

---

## What to take from it

- **Verification and identification are different problems and you must say which
  one you mean.** A per-comparison error rate is multiplied by the gallery size
  and then divided by the base rate. On Section 1's illustrative numbers — one
  false match in a million, a gallery of 100,000, one search in a thousand
  involving an enrolled person — about **99 of every 100 returned candidates are
  strangers**, with an algorithm nobody would call inaccurate.
- **The differentials are in false positives, and a false positive is the
  expensive error here.** NIST measured variation "by factors of 10 to beyond
  100" across demographic groups in 2019 — and also found vendors with no
  detectable differential, and that higher overall accuracy generally means
  smaller gaps. Quote the audit, the algorithm and the date, or quote no number.
- **Ask where the training faces came from before you ask how well the model
  performs.** A licence to copy a photograph is not consent from the person in
  it, and two of the field's most-used datasets were withdrawn once somebody
  looked. Building the largest of them is now prohibited in the EU rather than a
  research contribution.
- **The failure that reaches a courtroom is almost never the model.** It is the
  pipeline around it: a lead treated as an identification, and a confirmation
  sought from a witness who was shown the machine's own answer. That is Lesson
  1's leakage and Lesson 5's protocol discipline, with a person's liberty as the
  dependent variable.

One thread ties back to Lesson 4's supplementary reading, which ended on the
impossibility result: when two groups have different base rates, no classifier
can be calibrated within groups and have equal error rates across them. Face
identification is that theorem with the base rate made visible. The prior
probability that the person in front of the camera is on the watchlist is tiny,
differs between the populations that get photographed, and is set by who gets
policed rather than by anything in the data. You cannot tune your way out of it.
You can choose whether the number leaves the building as a lead or as a name —
and that choice is usually made by an engineer.

## Where to look next

| Resource | Type | Why read it |
|---|---|---|
| Grother, Ngan & Hanaoka, *FRVT Part 3: Demographic Effects*, NISTIR 8280 (2019) | Government evaluation | The measurement everyone cites and few read; the executive summary alone repays twenty minutes, and its warnings about how to report bias beat most papers' |
| Buolamwini & Gebru, "Gender Shades", *PMLR* 81 (2018) | Paper | The audit that redirected the field's attention; read it knowing it evaluates classification, not recognition |
| *R (Bridges) v Chief Constable of South Wales Police* [2020] EWCA Civ 1058 | Court judgment | Short, readable, and the clearest existing statement of what a lawful deployment would have to look like |
| Regulation (EU) 2024/1689 (AI Act), Article 5 | Primary legal text | The prohibitions in the words that bind, at `eur-lex.europa.eu` |
| Regulation (EU) 2016/679 (GDPR), Articles 4 and 9 with Recital 51 | Primary legal text | Where a photograph becomes biometric data, and what follows |
| Harvey & LaPlace, *Exposing.ai* | Investigative project | Dataset provenance traced image by image; find out whether your own photographs are in MegaFace |
| Buolamwini, *Unmasking AI* (2023) | Book | The audits from inside, including how the vendors and institutions responded |
