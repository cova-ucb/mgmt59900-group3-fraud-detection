# Model Card: PaySim Fraud-Scoring Model

**MGMT 59900 Group 3 | PaySim Fraud Detection | Armando, Phase E | Updated Aug 10 (post-redo)**

## Purpose

A fraud-scoring model that flags TRANSFER and CASH_OUT transactions for review, evaluated against the PaySim simulator's own rule-based fraud flag (`isflaggedfraud`) as the naive baseline, per the Data Quality Report's finding that this baseline catches only 16 of 8,213 real fraud cases dataset-wide. Training data is scoped to TRANSFER and CASH_OUT transactions specifically, since fraud occurs exclusively in these two types.

## Intended Use, Excluded Use & Oversight

Per the course's model-card framework (Week 6, Responsible ML in Practice):

- **Intended use:** prioritize TRANSFER and CASH_OUT transactions for fraud review during authorization, supporting a human-in-the-loop decision, such as approve, hold for step-up verification, or route to analyst review.
- **Excluded use:** must not be used to automatically decline or close an account without human review; must not be treated as the sole basis for a permanent fraud determination; must not be applied to CASH_IN, DEBIT, or PAYMENT transactions, since the model was trained and evaluated exclusively on TRANSFER/CASH_OUT and has no evidence base for the other three types.
- **Oversight:** human-in-the-loop for any decline or hold decision. This matches the course's own worked example for high-value fraud declines: given the financial stakes of an incorrect block, a person (which can include the customer, via a step-up verification prompt) should make the final call, not the model alone.

## Training Pattern vs. Production Deployment

This model was trained via **batch, notebook-based experimentation** (a SageMaker notebook instance), the appropriate pattern for the exploration/evaluation stages of the ML lifecycle. However, the underlying business decision, authorizing a TRANSFER or CASH_OUT, is a **real-time decision**, matching the milliseconds-to-seconds Latency stage already defined in the project's design canvas. A production version of this model would need to be deployed to an **online inference endpoint**, not run as a scheduled batch job, and would also pass through a formal **model registry** step (e.g., SageMaker Model Registry) before deployment; neither of these is in scope for this phase of the project. This notebook represents the Prepare/Train/Evaluate stages of the six-stage managed ML lifecycle. Register, Deploy, and Monitor remain future work.

## Data

Source: `paysim_db.curated_silver` (Phase C output). Training scope: TRANSFER and CASH_OUT transactions only (2,770,409 of 6,362,620 rows), since fraud does not occur in CASH_IN, DEBIT, or PAYMENT. This scoping matches the actual fraud population without discarding any of the 8,213 labeled fraud cases. 100% DQDL-validated, zero nulls.

**Features used (4):** `amount`, `txn_velocity`, `avg_amount_by_acct`, `amount_deviation`, all engineered in Phase C to capture per-account behavioral patterns.

**Excluded fields, and why:**
- `oldbalancedest`, `newbalancedest`: excluded per the Data Quality Report's leakage finding (99.29% of fraudulent TRANSFERs show a zero destination balance vs. 0.02% for legitimate ones; a simulation artifact, not a real fraud signal).
- `oldbalanceorg`, `newbalanceorig`: **tested and excluded** for the mirror-image reason on the sender side; see "Feature Leakage Discovery" below.
- `nameorig`, `namedest`: account identifiers, not predictive features; also excluded from the initial S3 read entirely for memory reasons (high-cardinality strings caused an out-of-memory kernel crash on the first training attempt).
- `step`, `step_day`: excluded as features to avoid the model simply memorizing the train/val/test time boundary rather than learning a generalizable pattern.
- `type`: not encoded/included in this iteration, a natural next step (see Limitations). Not needed for scoping, since training data is already filtered to TRANSFER/CASH_OUT only.
- `isflaggedfraud`: this is the baseline being evaluated against, not a model input.

## Feature Leakage Discovery: A Worked Example

An earlier version of this model's plan (documented in the Technical Checkpoint's data dictionary) said to include `oldbalanceOrg` and `newbalanceOrig` as features, on the idea that sender-balance information could help predict fraud. Including them raised validation AUCPR to 0.89 to 0.96, a huge jump from the original 4-feature model's roughly 0.065. This course warns that a result this strong should raise suspicion, not confidence. So instead of accepting it, the result was tested:

```python
fraud_drained = ((fraud_train['oldbalanceorg'] - fraud_train['amount']).abs() < 0.01).mean()
legit_drained  = ((legit_train['oldbalanceorg'] - legit_train['amount']).abs() < 0.01).mean()
```

**Result: 97.93% of fraudulent transactions fully drain the sender's account (`oldbalanceorg == amount`), compared to 0.00% of legitimate transactions.** This is nearly a perfect signal, the same kind of problem already found on the destination-balance side in the Data Quality Report. It is a quirk of the simulator, not something that would hold true in a real payment system. Both sender-balance fields were removed and the model was retrained on the original 4 features. This also means the Checkpoint's data dictionary (Table 2 / Section 4.4) had the wrong plan, while its Table 1 (which never listed these fields) was right all along.

## Train / Validation / Test Split

Time-based split on `step` (not random shuffling), so the model never trains on data from "after" its own evaluation period:

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1 to 500 | 2,645,779 | 0.2102% |
| Validation | 501 to 594 | 73,344 | 1.3607% |
| Test (held out, evaluated once) | 595 to 743 | 51,286 | 3.2251% |

**Note on the fraud-rate progression:** fraud rate rises sharply across later steps because transaction *volume* drops off in the later simulation period while fraud *count* stays roughly flat, the same pattern found in Phase D's hour-of-day query (fraud rate spikes during low-traffic windows). This is a real, documented characteristic of the dataset, not a split error.

**Note on scoping's effect on these rates:** scoping to TRANSFER/CASH_OUT roughly doubles the fraud rate at every split level compared to training on the full unscoped dataset (e.g., test fraud rate rises from 1.34% to 3.23%). This is not because more fraud exists; all 8,213 fraud cases remain, and the test set still contains roughly 1,654 of them either way. It is because the non-fraud denominator shrinks once transaction types that never contain fraud are removed.

## Class-Imbalance Handling

- **Logistic Regression:** `class_weight='balanced'` (automatic inverse-frequency weighting)
- **XGBoost:** `scale_pos_weight` set to the train-set imbalance ratio (approximately 474.8, lower than the unscoped run's approximately 950.2, since scoping raises the training fraud rate)

## Model Selection & Tuning

Two models trained: a logistic regression baseline, and XGBoost as the primary model.

XGBoost was tuned via a manual 4-configuration search (varying `max_depth`, `n_estimators`, `learning_rate`) evaluated against the **validation** set, not the test set, keeping the final test evaluation honest and unseen. `GridSearchCV` was deliberately not used: its default cross-validation shuffles data randomly across folds, which would let the model train on future transactions and validate on past ones within a fold, violating the time-based methodology chosen specifically to avoid that.

**Tuning result:** all 4 configurations landed within 0.0011 validation AUCPR of each other (0.0851 to 0.0862), the same plateau pattern found in the original (pre-scoping) tuning run, now confirmed a second time on independently corrected data. This indicates the 4-feature set has reached its performance ceiling regardless of population scoping; the constraint is the feature set, not model configuration. The best configuration (`max_depth=3, n_estimators=100, learning_rate=0.1`) was carried forward to final test evaluation.

## Scoped vs. Unscoped: A Side-by-Side Comparison

The redo changed two things at once: which transactions the model trains on (scoping to TRANSFER/CASH_OUT) and which features it uses (removing the leaky sender-balance fields). To make that change easy to see, here are both runs side by side:

| | Original run (unscoped) | This run (scoped, leakage-corrected) |
|---|---|---|
| Transaction types included | All 5 types | TRANSFER and CASH_OUT only |
| Rows used for training/testing | 6,362,620 | 2,770,409 |
| Test set fraud rate | 1.34% | 3.23% |
| Logistic Regression AUCPR | 0.2029 | 0.2109 |
| XGBoost (tuned) AUCPR | 0.1681 | 0.1858 |
| Does Logistic Regression beat XGBoost? | Yes | Yes, again |

**What this table does and does not show.** It does not mean the scoped model is simply "better." The two runs use different populations with different fraud rates, so their AUCPR scores are not directly comparable numbers. What the table does show is that the same core finding, logistic regression beating XGBoost, held up twice, on two different versions of the data. That repetition is the useful result, not the small change in the AUCPR numbers themselves.

## Final Results (Test Set, Evaluated Once)

These are this run's numbers only (scoped, leakage-corrected), the ones that should be used in the Report and presentation:

| Model | Precision | Recall | AUCPR | Fraud cases caught (of ~1,654) |
|---|---|---|---|---|
| `isflaggedfraud` baseline | 1.000 | 0.0048 | n/a | ~8 |
| Logistic Regression | 0.1583 | 0.4347 | **0.2109** | ~719 |
| XGBoost (tuned) | 0.0926 | 0.5266 | 0.1858 | ~871 |

**A note on comparing these numbers to the earlier run:** see the side-by-side comparison above. The short version: these numbers should not be described as "better than before," since the two runs used different data. They should be described as "confirms the same finding a second time."

## Three-Tier Decision-Threshold Policy

Following this course's framework for turning model scores into operational policy, the primary model's predicted probabilities on the test set were bucketed into three tiers: 0.00 to 0.30 approve, 0.30 to 0.70 manual review, 0.70 to 1.00 decline/escalate.

| Tier | Transactions | % of test set | Fraud caught | Fraud rate in tier |
|---|---|---|---|---|
| Approve | 7,262 | 14.2% | 100 | 1.38% |
| Review | 40,893 | 79.7% | 927 | 2.27% |
| Decline | 3,131 | 6.1% | 627 | 20.03% |

**The decline tier is well-concentrated:** a 20.03% fraud rate is 6.2 times the overall test-set baseline (3.23%), and only 6% of fraud (100 of 1,654 cases) slips through entirely unreviewed in the approve tier.

**The finding worth stating plainly: these thresholds are not operationally usable as chosen.** 79.7% of all transactions land in "review," which is operationally impossible for any real fraud team to process manually. This is a direct, first-party demonstration of this course's own capacity-constraint warning (its worked example: lowering a review threshold from 0.50 to 0.30 raised daily reviews from 1,000 to 4,500, against a team capacity of only 2,000). A natural next step, not pursued here given project timeline, would be deriving thresholds against an assumed review-team capacity (e.g., "the team can review 5% of daily volume") rather than the arbitrary round-number cutoffs used here.

## Key Finding

**Both models catch roughly 90 to 100 times more fraud than the naive rule-based baseline** (approximately 719 to 871 cases vs. approximately 8), the core result motivating the ML-based approach described in the Proposal.

**Logistic regression outperforms tuned XGBoost on AUCPR** (0.2109 vs. 0.1858), despite XGBoost catching more fraud cases at the default threshold. This is the second time this result has held: first on the original unscoped run, now again after correcting for scoping and leakage, making it a considerably more defensible claim than a single result would be. AUCPR is the more robust, threshold-independent metric for a severely imbalanced problem; with only 4 simple, non-leaky features, the added complexity of gradient-boosted trees does not translate into better ranking performance.

**The three-tier threshold policy surfaces a genuine operational constraint**, not just a modeling result. The model's own scores, run through reasonable-sounding round-number thresholds, would route 80% of transactions to manual review, an infeasible workload for any real fraud team. This is presented as a finding about thresholds-as-policy, not a flaw in the model.

## Limitations & Future Improvements

- **Feature set is minimal.** A "new/unfamiliar destination account" flag was scoped in Phase C but left as a placeholder pending a self-join that wasn't completed. Adding it, along with an encoded `type` feature, is the most promising next step for improving performance, more so than further model tuning, given the tuning search already showed diminishing returns twice.
- **`txn_velocity` is currently a batch-computed feature**, calculated once during Glue ETL against static historical data. This course's material names "recent transaction velocity" as the canonical example of a *streaming* feature; a production deployment would need to recompute it within seconds of each transaction, not as a static historical batch value, to actually support a real-time authorization decision.
- **Threshold policy is not yet capacity-aware.** The 0.30/0.70 thresholds were chosen as round numbers, not derived from an actual review-team capacity constraint, and the resulting 79.7% review-tier workload demonstrates why that matters. Re-deriving thresholds against an assumed daily review capacity is a natural next step.
- **Data drift, specifically.** This course's material distinguishes data drift (the input distribution changes), concept drift (the relationship between inputs and outcome changes), and data-quality failure (inputs go missing or wrong). The fraud-rate progression across train/validation/test, more than doubling at each stage, is a documented instance of **data drift**: the transaction-volume distribution genuinely shifts across the simulated time period. This is a real property of the dataset and a legitimate stress test of the model, not a flaw, but a production system would need explicit data-drift monitoring given how pronounced this shift is even within one simulated month.
- **No production monitoring plan implemented.** A deployed version of this model would need monitoring across this course's four-layer framework: data monitoring (feature distributions, missing values), model monitoring (accuracy and subgroup performance over time), operational monitoring (latency, inference cost), and business monitoring (whether flagged transactions actually reduce fraud losses, not just whether the model's statistics look healthy).
- **GridSearchCV not used**, for the time-series-leakage reason described above. A `TimeSeriesSplit`-based search was not pursued given the manual search's clear plateau result, confirmed twice.
- **No subgroup performance analysis.** This course's material emphasizes that an acceptable overall error rate can hide much higher error for a specific subgroup (e.g., the lecture's international-traveler fraud-review example: 4% overall false-decline rate vs. 11% for international travelers). A natural next step is evaluating precision/recall separately by transaction type (TRANSFER vs. CASH_OUT) rather than relying on the pooled metrics reported above.
