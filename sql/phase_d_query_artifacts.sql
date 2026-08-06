-- =============================================================
-- Phase D — Query Artifacts for Phillip's Dashboard & Query Plan
-- MGMT 59900 Group 3 | PaySim Fraud Detection
-- Run in Athena, database: paysim_db
-- All queries run against paysim_db.curated_silver
-- (Phase C's feature-engineered, DQDL-validated output —
-- NOT curated_transactions, which is Phase B's earlier bronze-layer
-- table and is not used for anything past Phase B.)
-- =============================================================

-- Sanity check: full row count (should be 6,362,620 — 100% of source
-- data passed all 8 DQDL rules, zero rows rejected)
SELECT COUNT(*) AS curated_silver_count FROM paysim_db.curated_silver;


-- (a) Fraud rate by transaction type
-- Answers: which transaction types carry fraud risk, and how much?
-- Feeds: Dashboard chart 1
-- Result: TRANSFER 0.77% fraud rate (4,097/532,909); CASH_OUT 0.18%
-- (4,116/2,237,500); zero fraud in CASH_IN/DEBIT/PAYMENT.
SELECT type, COUNT(*) AS txn_count, SUM(isFraud) AS fraud_count,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
FROM paysim_db.curated_silver
GROUP BY type
ORDER BY fraud_rate_pct DESC;


-- (b) Fraud by step-bucket (hour-of-day view; step = simulated hour)
-- Answers: does fraud cluster at certain times?
-- Feeds: Dashboard chart 2
-- Key finding: fraud COUNT stays roughly flat (~300-375/hour)
-- regardless of transaction volume, so fraud RATE spikes to ~22%
-- during low-traffic overnight hours (4-5) vs. under 0.1% at peak
-- hours (9, 20). Worth leading the dashboard with this.
SELECT step % 24 AS hour_of_day, COUNT(*) AS txn_count, SUM(isFraud) AS fraud_count
FROM paysim_db.curated_silver
GROUP BY step % 24
ORDER BY hour_of_day;


-- (c) TRANSFER/CASH_OUT-only view
-- Answers: scopes the data to the only transaction types where fraud occurs
-- Feeds: Armando's Phase E model training and this dashboard
-- Result: 2,770,409 rows (532,909 TRANSFER + 2,237,500 CASH_OUT)
CREATE VIEW paysim_db.transfer_cashout_only AS
SELECT * FROM paysim_db.curated_silver
WHERE type IN ('TRANSFER','CASH_OUT');

-- Verify the view
SELECT COUNT(*) FROM paysim_db.transfer_cashout_only;

-- Sample export for Phillip (cross-account handoff — see /evidence and
-- the data dictionary for the CSV exports actually shared)
SELECT * FROM paysim_db.transfer_cashout_only LIMIT 1000;


-- (d) isFlaggedFraud vs. actual isFraud
-- Answers: how effective is the simulator's own rule-based fraud flag?
-- Feeds: Dashboard chart 3; also feeds the Data Quality Report and the
-- model card's baseline comparison in Phase E.
-- Result: of 8,213 real fraud cases, isFlaggedFraud catches only 16
-- (recall ~0.19%), with zero false positives. This is the naive
-- baseline Armando's Phase E model needs to beat.
SELECT isFraud, isFlaggedFraud, COUNT(*) AS txn_count
FROM paysim_db.curated_silver
GROUP BY isFraud, isFlaggedFraud;
