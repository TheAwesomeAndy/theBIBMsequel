# Paper A against the research harness rulebook

2026-11-22. Source: `TheAwesomeAndy/myResearchAssitantHarness`, `docs/master-rule-framework.md`
and `docs/checks.md`, read in full. The `papercheck` engine was not executed in this
session, so every row below is a manual reading of the rule against the manuscript.

| Rule | Harness requirement | Paper A | Status |
|---|---|---|---|
| `title.length`, title rules | 8 to 13 words, declarative, no metaphor, colon subtitle allowed | "Per-Epoch Normalization Breaks Causality in EEG Decoding: Exact Leakage Identities and an Averaging Law" (13 words) | met |
| `abstract.length` | under 200 words | about 200 by a LaTeX-stripped count (was 247) | at the bound |
| `abstract.structure` | four sentences | ten sentences carrying the key numbers | not met, by choice. IEEE abstracts of this type report results. Author decision |
| `intro.contributions`, `intro.forward-refs` | itemized contributions, each with a section reference | four contributions in prose, one section reference | partly met. An itemized list costs about six lines |
| `intro.roadmap`, `intro.cliche-opening` | none | none | met |
| `methods.leakage` | subject-wise validation for human data | participant-grouped folds throughout | met |
| `methods.reproducibility` | seeds, versions, code | seeds and public code stated | met |
| `methods.future-tense` | past tense | present tense throughout the methods | not met. Mechanical to convert |
| `structure.sections`, `conclusion.*` | a Conclusion section | none. The paper ends on Scope and Future Directions | not met. Needs about eight lines and an equal cut |
| `limits.present` | one to three stated limitations with a pivot | "Scope and Future Directions" states scope positively, per the project voice directive, which bans the word the check looks for | conflict between the two rulebooks. Author decision |
| `ethics.statement` | IRB and consent statement for human data | none | not met. A submission blocker for TNSRE. The protocol number must come from the data custodian |
| `data.availability` | present | present | met |
| `discussion.compare` | citations in the discussion | present | met |
| `bib.uncited` | no unused entries | nine entries were unused, six of them from the target journal. Four are now cited where they apply, five remain unused | improved |
| Rule 2.5 journal selection | journals appearing two or more times in the reference list | TNSRE appears seven times, J. Neural Eng. three times | consistent with the venue ranking |
| Layer 1 rule 1 | one scoped phase at a time, stop for verification | this session ran several phases without stopping | departure, at the author's instruction to finish work without handing steps back |
