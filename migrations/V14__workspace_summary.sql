-- Add rolling summary to workspaces for context injection.
ALTER TABLE agent_workspaces ADD COLUMN summary TEXT;
