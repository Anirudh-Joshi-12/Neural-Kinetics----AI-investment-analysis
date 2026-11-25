-- Migration 003: Add is_exceptional column to reports table.
-- This migration adds the is_exceptional column which flags companies that
-- are exceptionally good -- all KPIs are top-tier AND ML score is Strong Buy
-- or better. This is a positive outlier, not a negative one.

ALTER TABLE reports ADD COLUMN IF NOT EXISTS is_exceptional BOOLEAN;

INSERT INTO _schema_migrations (filename) VALUES ('003_add_exceptional_column.sql')
ON CONFLICT (filename) DO NOTHING;
