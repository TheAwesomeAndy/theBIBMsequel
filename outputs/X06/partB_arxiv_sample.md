# X06 Part B: arXiv sample (40 papers, coded 2026-11-21)

Sampling rule and codes: experiments/X06/PROTOCOL.md. Candidates: arXiv API, newest first from 2025-12-27; entries that did not report a supervised EEG decoding experiment were skipped (2 of the first 42: a zero-shot classifier of ICA component images, 2511.?; an ERP estimation method). Coding read the sentences returned by a keyword screen for scaling terms (z-score, normaliz/standardiz, min-max, rescal, scaled, zero mean, unit variance, divided by) in each full text; UNSTATED means the screen returned no sentence that decides the input scaling.

Counts: UNSTATED 28, OWN 4, FIXED 3, TRAIN 3, NONE 1, PRE 1

| arXiv | Title | Code | Evidence | Note |
|---|---|---|---|---|
| 2512.22656 | Clinically Calibrated Machine Learning Benchmarks for Large-Scale Mult | UNSTATED | no deciding sentence found by the screen |  |
| 2512.22333 | Emotion classification using EEG headset signals and Random Forest | UNSTATED | no deciding sentence found by the screen |  |
| 2512.21170 | A Unified Framework for EEG Seizure Detection Using Universum-Integrat | UNSTATED | no deciding sentence found by the screen |  |
| 2512.20929 | Decoding Predictive Inference in Visual Language Processing via Spatio | UNSTATED | no deciding sentence found by the screen |  |
| 2512.20319 | Deep Learning Classification of EEG Responses to Multi-Dimensional Tra | UNSTATED | no deciding sentence found by the screen |  |
| 2512.19097 | DIVER-1: Scaling Intracranial EEG Foundation Models for Transferable R | FIXED | "normalized signals by dividing by 200 uV" | intracranial EEG |
| 2601.00020 | Personalized Spiking Neural Networks with Ferroelectric Synapses for E | UNSTATED | no deciding sentence found by the screen |  |
| 2512.18689 | Fusion of Multiscale Features Via Centralized Sparse-attention Network | UNSTATED | no deciding sentence found by the screen |  |
| 2512.18346 | Cognitive Inference based Feature Pyramid Network for Sentimental Anal | UNSTATED | no deciding sentence found by the screen |  |
| 2512.17775 | How Light Shapes Memory: Beta Synchrony in the Temporal-Parietal Corte | UNSTATED | no deciding sentence found by the screen |  |
| 2512.22172 | PaperNet: Efficient Temporal Convolutions and Channel Residual Attenti | TRAIN | "statistics computed from the training subset to avoid information leakage" |  |
| 2512.15250 | Leveraging Foundational Models and Simple Fusion for Multi-modal Physi | UNSTATED | no deciding sentence found by the screen |  |
| 2512.15246 | Enhancing Alzheimer's Detection through Late Fusion of Multi-Modal EEG | UNSTATED | no deciding sentence found by the screen |  |
| 2512.13806 | EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning  | OWN | "Each electrode within a window is then standardised independently" | trial-locked windows |
| 2512.13458 | SSAS: Cross-subject EEG-based Emotion Recognition through Source Selec | UNSTATED | no deciding sentence found by the screen |  |
| 2512.22146 | EEG-to-Voice Decoding of Spoken and Imagined speech Using Non-Invasive | OWN | "normalization was applied within each trial based on the minimum and maximum values" | trial-locked (task onset) |
| 2512.12273 | GRC-Net: Gram Residual Co-attention Net for epilepsy prediction | UNSTATED | no deciding sentence found by the screen |  |
| 2512.09524 | NeuroSketch: A Practical Design Recipe for Neural Decoding | TRAIN | "mean and standard deviation calculated across all extracted windows and time samples for each participant" | participant-wide, includes test windows |
| 2512.06725 | Decoding Motor Behavior Using Deep Learning and Reservoir Computing | UNSTATED | no deciding sentence found by the screen |  |
| 2512.16929 | BIONIX: A Wireless, Low-Cost Prosthetic Arm with Dual-Signal EEG and E | UNSTATED | no deciding sentence found by the screen |  |
| 2512.05528 | Decoding Selective Auditory Attention to Musical Elements in Ecologica | UNSTATED | no deciding sentence found by the screen |  |
| 2512.05500 | SSDLabeler: Realistic semi-synthetic data generation for multi-label a | UNSTATED | no deciding sentence found by the screen |  |
| 2512.02978 | Rethinking Generalized BCIs: Benchmarking 340,000+ Unique Algorithmic  | TRAIN | "feature normalization are confined strictly to the training folds" |  |
| 2512.00574 | GCMCG: A Clustering-Aware Graph Attention and Expert Fusion Network fo | OWN | each channel standardized by the mean and SD of that same denoised signal (Eq. 4) | trial-locked (4 s MI trials) |
| 2511.23142 | Adapting Neural Audio Codecs to EEG | FIXED | "clipped to +-200uV, normalized to the [-1, 1] range" |  |
| 2511.21940 | Deep Learning Architectures for Code-Modulated Visual Evoked Potential | FIXED | "raw EEG signals were first scaled to microvolts" | stimulus-locked (c-VEP) |
| 2511.20848 | NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activ | UNSTATED | no deciding sentence found by the screen |  |
| 2511.20570 | Gated Uncertainty-Aware Runtime Dual Invariants for Neural Signal-Cont | UNSTATED | no deciding sentence found by the screen |  |
| 2511.18940 | Geometry-Aware Deep Congruence Networks for Manifold Learning in Cross | NONE | covariance pipeline; "Future work will explore signal-level normalization" |  |
| 2511.20696 | Prototype-Guided Non-Exemplar Continual Learning for Cross-subject EEG | UNSTATED | no deciding sentence found by the screen |  |
| 2511.18571 | SAMBA: Toward an EEG Foundation Model via Differential Mamba and Coord | UNSTATED | no deciding sentence found by the screen |  |
| 2511.18294 | MultiDiffNet: A Multi-Objective Diffusion Framework for Generalizable  | UNSTATED | no deciding sentence found by the screen |  |
| 2511.17164 | Teager-Kaiser Energy Methods For EEG Feature Extraction In Biomedical  | UNSTATED | no deciding sentence found by the screen |  |
| 2511.15902 | EEG Emotion Recognition Through Deep Learning | UNSTATED | no deciding sentence found by the screen |  |
| 2511.15138 | Cross-Modal Consistency-Guided Active Learning for Affective BCI Syste | UNSTATED | no deciding sentence found by the screen |  |
| 2511.13954 | A Brain Wave Encodes a Thousand Tokens: Modeling Inter-Cortical Neural | OWN | "z-score normalization is applied independently to both the trial chunks" per channel | sliding windows over long trials |
| 2511.12394 | Multi-Domain EEG Representation Learning with Orthogonal Mapping and A | UNSTATED | no deciding sentence found by the screen |  |
| 2511.12073 | Informed Bootstrap Augmentation Improves EEG Decoding | PRE | "Each trial ... was z-scored using the pre-stimulus baseline (-200-0 ms)" | stimulus-locked |
| 2511.11940 | Learning the relative composition of EEG signals using pairwise relati | UNSTATED | no deciding sentence found by the screen |  |
| 2511.10935 | CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusi | UNSTATED | no deciding sentence found by the screen |  |
