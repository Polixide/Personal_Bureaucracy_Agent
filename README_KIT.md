# Kit for Claude Code (version 2: ReAct agent, adapting the existing repo)

1. Check out the most integrated branch of the repository, then copy everything in this folder into the repo ROOT
   (`CLAUDE.md` and the `.claude/` folder must be at the root). If the repo already has a `CLAUDE.md`, merge instead of overwriting.
2. Install Ollama, then run:
   ollama pull qwen3:8b        # or gemma4:e4b on 8 GB machines
   ollama pull bge-m3          # or qwen3-embedding:0.6b if slow
3. Start Claude Code in the repo root, run /skills to check that add-procedure, add-rule, add-tool,
   draft-rules-from-source, run-evals and document-module are listed, then paste KICKOFF_PROMPT.md as the first message.
4. Claude Code starts with an audit and a migration plan. Read docs/migration_plan.md before approving phase 0.
