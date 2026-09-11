---
name: commit-push
description: Use when writing a git commit message or preparing to commit and push changes to GitHub. Writes clear, conventional commit messages without adding any author or co-author attribution, then commits and pushes when the user explicitly asks to invoke this skill.
---

# Commit and Push

This skill governs how commits are written and pushed to GitHub. Invoke it
when the user asks to "commit", "commit and push", "push to github", or calls
this skill by name.

## Rules

1. **Never add anyone as author of the code.** Do not add `Author:` or
   `Co-authored-by:` trailers, `Signed-off-by:` lines, or any other authorship
   attribution to commit messages. The commit is authored by the person who
   runs it — do not attribute it to any tool, model, agent, or person.

2. **Write clear, proper commit messages.** Use the Conventional Commits
   format:

   ```
   <type>(<scope>): <subject>

   <body>
   ```

   - `type`: `feat`, `fix`, `refactor`, `docs`, `test`, `build`, `ci`,
     `perf`, `chore` (pick the one that best matches the change).
   - `scope`: optional; use it when a meaningful area exists (e.g.
     `feat(rust): ...`, `fix(core): ...`).
   - `subject`: imperative mood, <= 72 characters, no trailing period.
   - `body`: only when more context is genuinely needed. Explain *what* and
     *why*, never *how* line-by-line.

3. **Stage only intended files.** Before committing:
   - Run `git status` and `git diff --stat` (and `git diff` for unstaged
     review) to see what is tracked and changed.
   - Never commit secrets, API keys, credentials, private URLs, large raw
     data files, or generated build artifacts (`git status` will reveal them).
   - Stage only the files relevant to the change you are committing.

4. **Commit.** Use `git commit` with the message written per the rules above.
   Do not alter git hooks, amend, rebase, force-push, or create empty commits
   unless the user explicitly asks.

5. **Push only when told to.** Do not push on your own. Push happens only
   when the user explicitly instructs it (for example, by invoking this
   skill and saying push). When pushing: verify the remote and branch first,
   then `git push`, and report the result.

## Workflow

1. Inspect repository state: `git status`, `git diff`, `git log --oneline -10`.
2. Review the staged/unstaged changes and confirm nothing sensitive or
   unrelated is included.
3. Write the commit message per Conventional Commits with no authorship
   attribution.
4. Commit the change.
5. If (and only if) the user has asked to push: push to GitHub and report the
   push result. Otherwise, report that the commit is ready and ask before
   pushing.