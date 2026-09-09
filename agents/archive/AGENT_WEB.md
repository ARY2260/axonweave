# Documentation Web Agent Playbook

The docs site is a real application, not a static Markdown dump. Markdown is the source of truth; the React application renders it with navigation, code-copy controls, headings, links, lists, tables, blockquotes and inline code.

Before editing frontend code, read `docs-site/DESIGN.md` and `agents/web/DOCS_TOKENS.md`.

Every new page must:

1. have a stable route;
2. appear in the sidebar if public;
3. have a descriptive title and metadata;
4. expose headings for the TOC;
5. contain concrete examples and constraints;
6. avoid unsupported scientific claims;
7. remain keyboard accessible and mobile usable.
