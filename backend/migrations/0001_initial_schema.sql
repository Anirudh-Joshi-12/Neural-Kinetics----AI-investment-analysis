-- Migration 0001: Initial schema for reports and chunks tables.
-- Run this on a fresh database. Idempotent for CREATE EXTENSION / CREATE TABLE.

-- 1. Enable pgvector extension (must be superuser, runs first).
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Reports table: one row per uploaded annual report.
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_hash CHAR(64) UNIQUE NOT NULL,            -- SHA256 of PDF bytes, for idempotency
    pdf_path VARCHAR(512) NOT NULL,               -- local path to saved PDF
    original_filename VARCHAR(255),               -- user-provided filename

    kpis JSONB,                                   -- extracted KPIs (5 keys)
    score FLOAT,                                  -- ML score 0-100
    label VARCHAR(20),                            -- Strong Buy / Buy / Hold / Avoid / Strong Avoid
    confidence FLOAT,                             -- model confidence %
    shap_values JSONB,                            -- per-feature SHAP attributions
    is_ood BOOLEAN,                               -- OOD flag (true = low training coverage)
    ood_info JSONB,                               -- full OOD diagnostics (knn_pct, coverage, etc)

    brief TEXT,                                   -- Nemotron investment brief
    model_version VARCHAR(50) NOT NULL DEFAULT 'rf_v1',
    llm_model VARCHAR(100),
    llm_prompt_version VARCHAR(30),

    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending/processing/completed/failed
    error_message TEXT,
    processing_time_ms INT,
    human_label VARCHAR(20),
    labeled_at TIMESTAMP WITHOUT TIME ZONE,
    labeled_by VARCHAR(100),

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- 3. Chunks table: one row per text chunk + embedding.
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_type VARCHAR(20) NOT NULL DEFAULT 'other', -- fin_stmt/notes/risk/toc/other
    text TEXT NOT NULL,
    embedding VECTOR(4096),                         -- nemotron-3.5-embed-8b dimension

    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- 4. Indexes for common query patterns.
CREATE INDEX IF NOT EXISTS ix_reports_score_created ON reports(score, created_at);
CREATE INDEX IF NOT EXISTS ix_reports_model_version ON reports(model_version);
CREATE INDEX IF NOT EXISTS ix_chunks_report_id ON chunks(report_id);
CREATE INDEX IF NOT EXISTS ix_chunks_report_type ON chunks(report_id, chunk_type);

-- 5. Migration tracking table.
CREATE TABLE IF NOT EXISTS _schema_migrations (
    filename VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- 6. Record this migration as applied.
INSERT INTO _schema_migrations (filename) VALUES ('0001_initial_schema.sql')
ON CONFLICT (filename) DO NOTHING;