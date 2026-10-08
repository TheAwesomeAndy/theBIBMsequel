Dear Editor,

We submit "The Weight of Nothing: Missing Electrodes, Empty Baselines, and the Edge of Stability in Spiking Reservoir Encoding of Affective EEG" for consideration as a regular paper. The paper measures how far the representation of nothing moves results in affective EEG decoding: the value that replaces a missing electrode, the origin of a feature coordinate, the pre-stimulus baseline after per-epoch normalization, and the operating point of a fixed spiking reservoir. On the SHAPE cohort (211 subjects) and a public IAPS cohort (228 subjects), moving the fill of dropped channels changes the dropout score while clean predictions stay fixed, per-epoch z-scoring lets the 200 ms before stimulus onset classify affect at 0.540 balanced accuracy, and past the measured edge of stability a silent electrode costs the reservoir several times more accuracy than below it.

These choices are rarely reported, yet they change measured rankings and, in one case, the reading of a published result. The manuscript extends our IEEE BIBM 2026 Doctoral Forum paper S60204, "On the Edge of Stability: Spiking Reservoir State-Space Encoding of Affective EEG." Rerunning that study with analysis windows measured from true onset shows that its reservoir window read the normalization offset rather than evoked dynamics, and the paper states this correction. The new content is the SHAPE rerun, the external cohort, two propositions with direct tests, ten-draw operating-point sweeps judged by a decision rule fixed in advance, spherical-spline repair, recorded trial-count noise, and EEGNet recipe variation. Every number is generated from committed analysis code.

IEEE Transactions on Neural Systems and Rehabilitation Engineering publishes work on how EEG decoders are evaluated and how they tolerate electrode loss and noise. Our reporting recommendations, namely fill sweeps with feature centeredness, pre-stimulus decoding under the chosen normalization, and operating points across reservoir draws, are aimed at readers who build and compare such decoders.

Sincerely,
Andrew A. Lane
Department of Electrical and Computer Engineering, Stony Brook University
