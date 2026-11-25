-- Migration 0002: Add OOD (out-of-distribution) columns to reports table.
-- This migration adds the honesty layer columns that expose whether the
-- model's confidence is trustworthy for the given input KPI profile.

-- 1. Add is_ood column (nullable boolean - NULL = not yet evaluated).
ALTER TABLE reports ADD COLUMN IF NOT EXISTS is_ood BOOLEAN;

-- 2. Add ood_info column (nullable JSONB - full diagnostics).
ALTER TABLE reports ADD COLUMN IF NOT EXISTS ood_info JSONB;

-- 3. Record this migration as applied.
INSERT INTO _schema_migrations (filename) VALUES ('0002_add_ood_fields.sql')
ON CONFLICT (filename) DO NOTHING;