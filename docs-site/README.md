# AxonWeave Documentation Site

This directory contains the actual browser documentation application. Markdown files under `content/` are the content source; React renders them into a documentation UI.

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

The build emits a static site and copies `index.html` to `404.html` so direct routes work on GitHub Pages.

## Deployment

`.github/workflows/docs-pages.yml` builds and deploys the site to GitHub Pages after the repository's Pages environment is enabled.
