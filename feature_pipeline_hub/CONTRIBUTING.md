# Contributing Guidelines

## Git Workflow

### Branch Strategy

- **All work happens on the current branch** — no worktrees or parallel branches
- Changes are committed to the active branch as they progress
- Keep a single, focused context for iterative development

### Commit Policy

- **Manual commits only** — commits are made explicitly by the user, never automatically
- Claude will stage and prepare changes but **waits for your instruction** to commit
- Each commit should represent a logical, reviewable unit of work
- Commit messages should be clear and descriptive

### Workflow Steps

1. **Development**: Make changes directly on the current branch
2. **Review**: Code changes are staged and ready for inspection
3. **Commit**: User explicitly instructs when to commit (e.g., "commit this now")
4. **Message**: User provides commit message or Claude drafts one for approval

### Best Practices

- Review staged changes before committing
- Use atomic commits — each commit should be self-contained
- Keep commit messages concise and meaningful
- Push only when explicitly requested

---

**Note**: This project prioritizes clarity and control over automation. Claude respects the user's ownership of the repository history.

