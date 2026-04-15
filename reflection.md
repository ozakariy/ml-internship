## Task 1 — Biomarker Discovery

### Time spent
- **Part A — Exploratory Data Analysis:** ~3 days
- **Part B — Data Processing:** ~2 days
- **Part C — Discriminatory Analysis:** ~2 days

Overall, Task 1 took about **7 days**. This felt reasonable because Part A was broad and exploratory, while Parts B and C required more decision-making and iteration.

### Perceived difficulty
- **Part A:** medium to easy conceptually, but long in practice
  The main difficulty was not the individual computations themselves, but the amount of ground that had to be covered carefully. I had to audit the inputs, reconstruct the acquisition structure, inspect sample-level signal behavior, analyze QC stability, contamination, detection support, standards, redundancy, and technical-vs-biological variation. The work was manageable, but it required patience and a lot of figure-by-figure interpretation.

- **Part B:** difficult
  This was the hardest part for me because it required turning exploratory observations into a defensible filtering workflow for biological data. The main challenge was deciding how strict to be without removing potentially meaningful biological signal. In particular, choosing the detection threshold, separating QC from dQC, evaluating transformations, reviewing unusual biological samples, and introducing contamination and D-ratio as supporting criteria required more judgment than Part A.

- **Part C:** medium hard
  This part was easier than Part B once the filtered dataset was stable, but it was still challenging because I needed to combine statistics and machine learning in a disciplined way. The most important difficulty was avoiding overclaiming results in a highly redundant feature space. I therefore had to move from simple univariate rankings to a redundancy-aware, stability-aware analysis with cross-validated linear models and tiered biomarker prioritization.

### What worked well
A few things worked particularly well during Task 1:

- Building the analysis progressively, starting from data auditing and technical diagnostics before moving to filtering and discrimination.
- Re-running the operational analysis on **batch 1 only** after the global EDA showed that batch 2 was not appropriate for downstream biological discovery.
- Keeping the pipeline reproducible by saving intermediate outputs and turning key decisions into explicit helper functions instead of ad hoc notebook logic.
- Adding redundancy-aware analysis in Part C, which made the final biomarker shortlist much more defensible than a simple ranked list of correlated LC-MS peaks.

### Challenges encountered and how I addressed them
The main challenge was distinguishing **technical instability** from **true biological variation**. In this dataset, many features were detected consistently, but the experiment also showed batch-related effects, QC/dQC differences, and heavy redundancy among retained LC-MS features.

I addressed this by:
- restricting the operational workflow to **batch 1**, which was the only batch containing usable biological samples and the most coherent technical behavior;
- using **QC-only** variability as the main technical stability reference and treating **dQC** as a secondary concentration-sensitivity check;
- fixing a positive detection threshold instead of using zero, in order to avoid counting trace blank signals as meaningful detections;
- selecting a transformed view that improved comparability across samples before doing downstream discrimination;
- adding contamination checks, D-ratio summaries, threshold-ablation analysis, and redundancy clustering so that the final retained set remained auditable and justified.

Another challenge was that the top discriminatory features in Part C were often not independent from one another. To avoid presenting one cluster of related peaks as many separate biomarkers, I introduced representative features and cluster-aware summaries, and then combined univariate evidence with repeated cross-validated linear modeling.

### Feedback on the assignment structure
I thought Task 1 was well designed because it tested several different skills: exploratory analysis, scientific reasoning, filtering decisions, and downstream discrimination. I also liked that the assignment did not reduce the task to just fitting a model, but required technical interpretation and justification.

The hardest aspect was that glycobiology and LC-MS data have a domain-specific structure that is not immediately intuitive for someone coming from a more general machine learning background. That made Part B especially demanding. At the same time, this was also one of the most interesting parts of the assignment, because it forced me to think more carefully about how preprocessing decisions affect biological conclusions.

## Task 2 — Biomarker Embedding

**Time spent:** 3 days
**Perceived difficulty:** Medium

Task 2 focused on building an interpretable glycan embedding workflow that could place the discovered glycans into a broader glycobiology context using the provided `glycan_list`, `df_glycan`, `glycan_binding`, and `N_glycans_df` resources. The main goal was not only to obtain a useful representation space, but also to justify each modeling choice and show evidence that the final embedding captured meaningful structural and contextual relationships. This is also reflected in the Task 2 code structure, which I organized as a stepwise pipeline from source auditing and canonicalization to embedding evaluation and final enrichment. :contentReference[oaicite:1]{index=1}

I started by auditing all Task 2 sources to understand sequence overlap, metadata coverage, and the amount of direct support available for the discovered glycans. This showed that the discovered set had only partial exact overlap with the external glycowork resources, which meant I could not rely on exact matching alone for interpretation. During this stage, I also found two issues that required correction before modeling: empty list-like metadata fields were being counted as if they contained information, and exact overlap between the discovered glycans and the control/reference resources was sensitive to notation mismatch. To address this, I created a canonical master table with both strict sequence keys and more relaxed topology-based keys, and I normalized metadata into a cleaner representation.

What worked well in this task was the structure-first modeling strategy. Before considering anything more complex, I built interpretable baseline feature spaces based on glycan composition and sequence tokens. These baselines were stronger than expected, especially on the N-glycan control task, where they already showed very high neighborhood purity. This was important because it justified not jumping immediately to graph-based or deep approaches. Instead, I moved to a controlled late-fusion strategy where I combined structural features with metadata and binding information and compared several candidate embeddings through explicit validation tasks. I found that the sequence-token representation remained the strongest pure structural baseline, while the combined `structure_metadata_binding` embedding gave the best overall contextual neighborhoods and was therefore the best choice for downstream interpretation.

The most important challenges in Task 2 were data quality and interpretation discipline. A small number of rows still behaved like metadata contamination rather than true glycan sequences, and those rows could distort nearest-neighbor retrieval if left untreated. I addressed this by adding a stricter plausibility filter before the final interpretation stage. Another challenge was that the Task 1 to Task 2 handoff was not fully resolved at the glycan level, so I could not honestly claim a complete end-to-end biomarker ranking across both tasks. I therefore kept the final Task 2 ranking conservative and described it as a contextual prioritization of discovered glycans rather than a definitive biomarker hierarchy. I think this was the right compromise because it preserved the integrity of the analysis instead of forcing weak assumptions.

In the final stages, I projected the discovered glycans into the selected embedding space and built a neighborhood-based interpretation workflow. For each discovered glycan, I summarized structural neighbors, N-glycan support, contextual metadata, disease-related context, and protein-related support inferred from the neighborhood. I also added family-level labels, motif-proxy summaries, robustness checks across embeddings, and two secondary relational views: a glycan similarity network and a glycan–protein context network. These additions made the final output more useful and better connected to the assignment objective of enriching the discovered glycans with biologically meaningful context.

Overall, I think Task 2 went well because the workflow stayed disciplined: I started with simple, interpretable baselines, used explicit validation tasks to compare representations, selected the final embedding based on evidence, and kept the final biological interpretation cautious where support was weak. The final deliverables were an enriched discovered-glycan table and an embedding evaluation report table, which together made the Task 2 analysis more structured and reproducible.

**Feedback on the assignment structure:** I found Task 2 interesting and well designed because it encourages a real machine learning workflow rather than just building a model directly. In particular, I appreciated the emphasis on simpler baselines first and on validating representation quality before interpretation. The main ambiguity for me was the handoff between Task 1 and Task 2: because the discovered Task 1 features were not fully mapped to glycans at the sequence level, some parts of the final prioritization had to remain provisional. A slightly more explicit handoff file or a clearer mapping expectation would make the transition between the two tasks easier and would strengthen the final integrated interpretation.
