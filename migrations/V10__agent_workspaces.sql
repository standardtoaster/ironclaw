-- Agent workspaces: persistent, automatically-routed conversation contexts.
-- Each workspace wraps a long-lived conversation that can be resumed.
-- The system routes incoming requests to the best-matching workspace
-- via embedding similarity on the topic field.

CREATE TABLE agent_workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    topic_embedding VECTOR,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active',
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT now(),
    turn_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_workspaces_user_id ON agent_workspaces(user_id);
CREATE INDEX idx_agent_workspaces_user_status ON agent_workspaces(user_id, status);

CREATE TRIGGER update_agent_workspaces_updated_at
    BEFORE UPDATE ON agent_workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Link routines to workspaces so triggered routines resume the right context.
ALTER TABLE routines ADD COLUMN workspace_id UUID REFERENCES agent_workspaces(id) ON DELETE SET NULL;
