Build one lesson of Technologies for Artificial Intelligence.

**This command has been replaced by the `new-lesson` skill.** Invoke that
instead — it carries the same artefact specifications plus the phase order, the
subagent briefs, and the review steps this command lacked.

```
/new-lesson {NN}
```

If the skill is not offered, read `.claude/skills/new-lesson/SKILL.md` directly
and follow it.

---

## Why it moved

This command wrote the handout first and derived everything else from it. That
order is wrong, and the repository has the evidence: a handout written before
the code runs has to invent its numbers, and prose written ahead of execution
was wrong four times across lessons 3 to 5.

The skill uses the order that works — data and executed notebooks, then the
handout against those numbers, then the remaining artefacts in parallel, then a
review that executes rather than reads. `CLAUDE.md`, under "Building a lesson
with subagents", says why.

It also adds what this command had no notion of: `Resources/` as a required
artefact, `Docs/worked_examples.py` as the arithmetic check, and a phase
detector, so a lesson interrupted halfway is resumed from the filesystem rather
than from memory.

```bash
python .claude/skills/new-lesson/scripts/lesson_state.py {NN}
```
