-- =============================================================
-- Phase B — Ingestion & Storage
-- MGMT 59900 Group 3 | PaySim Fraud Detection
-- Run in Athena, database: paysim_db
-- =============================================================

-- Create the database
CREATE DATABASE IF NOT EXISTS paysim_db;

-- Raw external table over the PaySim CSV in S3 (raw/ zone)
CREATE EXTERNAL TABLE paysim_db.paysim_raw (
  step INT,
  type STRING,
  amount DOUBLE,
  nameOrig STRING,
  oldbalanceOrg DOUBLE,
  newbalanceOrig DOUBLE,
  nameDest STRING,
  oldbalanceDest DOUBLE,
  newbalanceDest DOUBLE,
  isFraud INT,
  isFlaggedFraud INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"')
LOCATION 's3://mgmt59900-g3-paysim/raw/'
TBLPROPERTIES ('skip.header.line.count'='1');

-- Verify the raw table reads correctly
SELECT * FROM paysim_db.paysim_raw LIMIT 10;

-- CTAS: curated, partitioned Parquet table (Phase B bronze-layer conversion)
-- This is the basic CSV-to-Parquet conversion, no feature engineering,
-- no DQ filtering. Superseded for downstream analytics by Phase C's
-- curated_silver table (see phase_c note below) — kept here as an
-- intentional intermediate artifact, tied to the Phase B scan-comparison
-- evidence (470.68 MB raw scan vs. 1.72 MB Parquet scan, ~273x reduction).
CREATE TABLE paysim_db.curated_transactions
WITH (
  format = 'PARQUET',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['step_day'],
  external_location = 's3://mgmt59900-g3-paysim/curated/'
) AS
SELECT
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud,
  CAST(step / 24 AS INT) AS step_day
FROM paysim_db.paysim_raw;

-- Scan-comparison evidence: same aggregate query against raw vs. curated
-- (compare "Data scanned" in the Athena results panel for each)
SELECT type, COUNT(*) AS txn_count, SUM(isFraud) AS fraud_count
FROM paysim_db.paysim_raw
GROUP BY type;

SELECT type, COUNT(*) AS txn_count, SUM(isFraud) AS fraud_count
FROM paysim_db.curated_transactions
GROUP BY type;

-- NOTE: paysim_db.curated_silver (Phase C's fully processed,
-- feature-engineered, DQDL-validated table) is built by the Glue ETL
-- job (see /glue), not by SQL — it is NOT created in this file.
-- All downstream queries (Phase D and later) use curated_silver,
-- not curated_transactions.
