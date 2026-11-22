# Journal selection and evidence matrix

Updated 2026-11-22. Source labels: **[official]** a publisher or society page read this
session or the last; **[third-party]** an aggregator; **[prior session]** found in the
2026-11-21 session and not re-read today; **[inference]** a judgment, not a fact.
No acceptance probability is given anywhere in this file, because none can be verified.

## Paper A: normalization leakage

| | IEEE TNSRE | J. Neural Eng. (IOP) | IEEE JBHI | IEEE TBME |
|---|---|---|---|---|
| Scope fit | Neural engineering and BCI methods. Two preprocessing-sensitivity papers from this journal are already cited (Robbins 2020, Del Pup 2025) [official, prior session] | Lists neural signal processing and neuromorphic engineering in scope [third-party copy of scope]. Published a 2025 replicability critique of speech-imagery decoding [search result] | Health informatics. Decoding-evaluation fit is indirect [inference] | "Novel engineering methods with demonstrated biomedical significance" [official]. A pitfall paper has to argue significance harder here [inference] |
| Impact metric | JIF 5.2 [prior session] | about 4.0, 2025 JCR [third-party] | not re-verified today | 4.6, 2025 [third-party] |
| Acceptance evidence | about 40%, editor-reported [prior session, not re-verified] | none verified | none verified | none verified |
| Speed | not verified | median 151 days received to accepted, from PubMed dates [third-party] | not verified | not verified |
| Length | overlength charges above 8 pages [official] | no fixed page limit known to me [unverified] | limit 14 pages, charges from page 9 [official] | charges above 8 pages, $250 per page for the first two [official] |
| Cost | fully open access, APC up to $2,160 [DOAJ] | hybrid | hybrid, OA $2,800 [official list] | hybrid, OA $2,495 [official] |
| Main reviewer risk | "obvious algebra" and "rehabilitation relevance" | "one affective cohort plus ERP CORE" | scope | significance bar |

**Ranking: 1. TNSRE, 2. J. Neural Eng., 3. JBHI.** [inference]
TNSRE stays first: it is the only venue with any acceptance evidence, it is IEEE and
open access, and it has published the two closest methodological neighbours. The change
from the last session is the fallback. J. Neural Eng. moves above JBHI because it
publishes decoding-pitfall papers and names neuromorphic engineering in its scope.
One conflict to settle: the fill-rule paper was earlier planned for J. Neural Eng. Two
papers at one journal is allowed, and the second would cite the first.

Current length is about 9.2 pages against the 8-page threshold, so one to two overlength
pages. The per-page rate for TNSRE was not found today.

## Paper B: reference centering

| | IEEE TBME | IEEE TNSRE | J. Neural Eng. | IEEE SPL |
|---|---|---|---|---|
| Lineage | Euclidean alignment (He and Wu 2020), Riemannian transfer (Zanini 2018), Riemannian Procrustes (Rodrigues 2019) all appeared here | Publishes cross-subject transfer for BCI [inference from general knowledge] | Publishes transfer and calibration work [inference] | Rejected May 2026 |
| Fit of the new framing | A law for how many unlabeled reference observations an alignment needs, compared against EA on public data, sits directly in this lineage [inference] | Good | Good | A letter cannot hold the derivation, five paradigms, and the alignment comparison |
| Length | 8 pages before charges [official] | 8 pages before charges [official] | flexible [unverified] | 4 pages plus references |

**Ranking: 1. IEEE TNSRE, submitted after Paper A, 2. J. Neural Eng., reach option IEEE TBME.** [inference]
X12 is complete. The reference-size law holds on five public paradigms (mean absolute
error 0.004 over 30 points) and the break-even size is located correctly in all five.
That makes the paper a general methods result. The same run shows that single-trial
gains are small, 0.000 to 0.036. TBME asks for demonstrated biomedical significance
[official], and a referee there can fairly call a 1 to 4 point gain modest. TNSRE and
J. Neural Eng. weigh the methodological rule more. TBME stays as the reach option
because the alignment lineage lives there. SPL is not recommended again: the paper is
now five pages with a derivation, two datasets, and an alignment comparison.
Papers A and B should not reach TNSRE in the same month, since the same associate
editors would see two ERP CORE methodology papers from one group.
