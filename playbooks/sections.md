# Section micro-rule checklists

Each item is tagged with the enforcing check id in backticks, or
`(agent)` / `(human)` when it is not deterministically checked.

## Title
- [ ] 8-13 active words `title.length`
- [ ] not a question `title.question`
- [ ] no generic filler opener `title.filler`
- [ ] no pun, metaphor, or lyric (agent)
- [ ] a colon subtitle is allowed (human)

## Abstract
- [ ] present `abstract.present`
- [ ] under 200 words `abstract.length`
- [ ] four sentences: problem, motivation, achievement, implications `abstract.structure`
- [ ] no citations `abstract.citations`

## Keywords
- [ ] 4-6 discoverable terms `keywords.count`

## Introduction
- [ ] concrete hook, no cliche opening `intro.cliche-opening`
- [ ] compound research gap, two or more archetypes `intro.gap`
- [ ] no roadmap paragraph `intro.roadmap`
- [ ] itemized crunchy contributions `intro.contributions`
- [ ] each contribution forward-references its section `intro.forward-refs`

## Methods
- [ ] present `methods.present`
- [ ] contemporary baselines `methods.baseline`
- [ ] subject-wise validation, no leakage `methods.leakage`
- [ ] reproducibility details `methods.reproducibility`
- [ ] perturbation stress tests `methods.robustness`
- [ ] past tense `methods.future-tense`
- [ ] vendors, versions, controls listed (human)

## Results
- [ ] facts only, no speculation `results.speculation`
- [ ] "significant" only with a reported test `lang.significant`

## Discussion (RCI matrix)
- [ ] present `discussion.present`
- [ ] Restate the result (agent)
- [ ] Compare with the literature `discussion.compare`
- [ ] Explain the finding (agent)
- [ ] Interpret with cautious verbs `discussion.hedging`
- [ ] no absolute claims `claims.overclaim`

## Figures and tables
- [ ] every float captioned `floats.caption`
- [ ] every float labelled `floats.label`
- [ ] every float referenced `floats.referenced`
- [ ] introduced before it appears `floats.ref-before`
- [ ] grayscale- and colorblind-safe, no color-only references `floats.color-only`

## Related work
- [ ] postponed to late in the paper `related.position`
- [ ] generous credit, no take-downs `related.hostile`
- [ ] synthesized, not a laundry list `related.laundry-list`
- [ ] critical comparison matrix vs. strongest prior work (human)

## Conclusion and limitations
- [ ] not a copy of the abstract `conclusion.abstract-copy`
- [ ] 1-3 limitations stated `limits.present`
- [ ] each limitation has a defense or pivot `limits.pivot`
