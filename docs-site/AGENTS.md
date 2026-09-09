# AGENTS.md — AxonWeave Documentation Site (Web Agent Contract)

This file governs AI agents working inside `docs-site/`. The project-wide contract lives at the repository root in `AGENTS.md`; read it first.

## Mission

The docs site is a real application, not a static Markdown dump. Markdown is the source of truth; the React application renders it with navigation, code-copy controls, headings, links, lists, tables, blockquotes and inline code.

## Before editing frontend code

Read, in order:

1. Root `AGENTS.md` (project-wide contract)
2. `DESIGN.md` in this directory — design tokens (YAML front matter) and design rationale
3. Repository-root `CODE_TOKENS.md` for cross-cutting library identifiers that may appear in examples

## Every new page must

1. Have a stable route.
2. Appear in the sidebar if public.
3. Have a descriptive title and metadata.
4. Expose headings for the TOC.
5. Contain concrete examples and constraints.
6. Avoid unsupported scientific claims.
7. Remain keyboard accessible and mobile usable.

## Design tokens

All visual values (colors, typography, radii, spacing, component tokens) are defined in `DESIGN.md` in this directory. Never invent a new visual value in component code; derive it from a token in `DESIGN.md`. If a token is missing, add it there first, then run:

```bash
npm run design:lint
```

## Content tokens

The documentation site uses semantic content tokens so AI agents can extend it without inventing a new visual system per page.

| Token | Required behavior |
|---|---|
| `DOC-HERO` | page title + one-sentence purpose + primary navigation action |
| `DOC-NOTE` | explanatory note, never a marketing slogan |
| `DOC-WARN` | constraint, limitation or scientific caution |
| `DOC-API` | API reference section |
| `DOC-CODE` | copyable, syntax-highlighted code block |
| `DOC-TOC` | generated page table of contents |
| `DOC-SEEALSO` | related pages |
| `DOC-SOURCE` | provenance/source link |
| `DOC-LEGAL` | legal/compliance content |

Content rules: concrete nouns, explicit prerequisites, exact commands, no generic buzzwords, no unsupported claims.
