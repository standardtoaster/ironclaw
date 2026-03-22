-- Full-text search index on conversation messages.
-- Enables memory_search to find content from past conversations,
-- not just explicitly saved memory documents.
ALTER TABLE conversation_messages
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_conversation_messages_fts
  ON conversation_messages USING gin(content_tsv);
