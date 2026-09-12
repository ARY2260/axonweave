// Render verification: reproduce the production markdown pipeline exactly
// (normalize transforms -> marked -> DOMPurify config) and fail on:
//   1. raw markdown leaking into rendered output (bold/italic/headings/links/
//      tables/callout fences as literal text)
//   2. callout transforms that consumed content incorrectly
//   3. internal doc links pointing at unregistered page slugs
// Run via `npm run verify:render`. CI fails the docs deploy when this errors.
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { JSDOM } from 'jsdom';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const contentDir = join(root, 'content');
const files = readdirSync(contentDir).filter(f => f.endsWith('.md'));

marked.setOptions({ gfm: true, breaks: false });
const window = new JSDOM('').window;
const purify = DOMPurify(window);

// The exact transforms from src/main.tsx (kept in sync; this script's CI
// failure indicates either a content bug or a renderer divergence).
const calloutBody = (body) => marked.parse(body).trim();
const normalize = (md) => md
  .replace(/^:::DOC-NOTE ?\n?([\s\S]*?)\n:::/gm, (_m, b) => `<div class="callout callout-note"><p class="callout-title">Note</p>\n${calloutBody(b)}\n</div>`)
  .replace(/^:::DOC-WARN ?\n?([\s\S]*?)\n:::/gm, (_m, b) => `<div class="callout callout-warn"><p class="callout-title">Constraint</p>\n${calloutBody(b)}\n</div>`)
  .replace(/^:::DOC-TIP ?\n?([\s\S]*?)\n:::/gm, (_m, b) => `<div class="callout callout-tip"><p class="callout-title">Tip</p>\n${calloutBody(b)}\n</div>`);
const render = (md) => purify.sanitize(marked.parse(normalize(md)), { ADD_ATTR: ['target', 'rel', 'class'] });

// Link rewrite from src/main.tsx: content-relative .md links become clean
// extension-less SPA routes before the HTML is handed to the renderer.
const BASE_PREFIX = '/axonweave';
const rewriteDocLinks = (html) => html.replace(
  /href="([A-Za-z0-9_-]+)\.md(#[^"]*)?"/g,
  (_m, slug, hash) => `href="${BASE_PREFIX}/${slug}${hash ?? ''}"`,
);
const renderFull = (md) => rewriteDocLinks(render(md));

// Page registry slugs from src/main.tsx (source of truth for routes).
const mainTs = readFileSync(join(root, 'src', 'main.tsx'), 'utf8');
const slugs = new Set([...mainTs.matchAll(/\{slug:'([^']+)'/g)].map(m => m[1]));

const stripCodeBlocks = (html) => html
  .replace(/<pre>[\s\S]*?<\/pre>/g, '')
  .replace(/<code>[\s\S]*?<\/code>/g, '');

const LEAK_PATTERNS = [
  [/^#{1,6} /m, 'raw heading'],
  [/\*\*[^*\n]+\*\*/, 'raw bold **text**'],
  [/(?<![\w*])\*[^*\n]+\*(?![\w*])/, 'raw italic *text*'],
  [/`[^`\n]+`/, 'raw inline code'],
  [/\[([^\]]+)\]\(([^)]+)\)/, 'raw markdown link'],
  [/^\|.+\|\n\s*\|[-| :]+\|/m, 'unparsed table'],
  [/^:::DOC-/m, 'unprocessed callout fence'],
  [/^\s*[-*] \[[ x]\] /m, 'task list syntax'],
];

let failures = 0;
let callouts = 0;
let docLinks = 0;
let inlineCode = 0;

for (const f of files) {
  const md = readFileSync(join(contentDir, f), 'utf8');
  callouts += (md.match(/^:::DOC-/gm) || []).length;  const html = renderFull(md);
  const prose = stripCodeBlocks(html);
  inlineCode += (html.match(/<code>/g) || []).length;

  // 1. Raw markdown leaks (checked outside code blocks only).
  for (const [re, label] of LEAK_PATTERNS) {
    if (re.test(prose)) {
      console.error(`FAIL ${f}: ${label} leaked into rendered output`);
      failures++;
    }
  }

  // 2. Internal doc links must have been rewritten to clean routes that
  //    target registered slugs (no .md, no dead targets).
  for (const m of prose.matchAll(/href="([A-Za-z0-9_-]+)\.md(#[^"]*)?"/g)) {
    console.error(`FAIL ${f}: doc link not rewritten to clean route: ${m[0]}`);
    failures++;
  }

  const internal = new RegExp('href="(?!https?:|#|mailto:)(?:[./]*axonweave)?/([A-Za-z0-9_-]*)(#[^"]*)?"', 'g');
  for (const m of prose.matchAll(internal)) {
    const target = m[1] || 'index';
    if (!slugs.has(target)) {
      console.error(`FAIL ${f}: internal link targets unknown slug '${target}'`);
      failures++;
    }
    docLinks++;
  }
}

// 3. Callout transform integrity: every fence must survive as a callout div.
for (const f of files) {
  const md = readFileSync(join(contentDir, f), 'utf8');
  const html = renderFull(md);
  const opened = (md.match(/^:::DOC-/gm) || []).length;
  const rendered = (html.match(/class="callout callout-/g) || []).length;
  if (opened !== rendered) {
    console.error(`FAIL ${f}: ${opened} callout fences but ${rendered} rendered callout divs`);
    failures++;
  }
}

console.log(`render check: ${files.length} pages, ${callouts} callouts, ${docLinks} internal links, ${inlineCode} code spans`);
if (failures) {
  console.error(`render check FAILED with ${failures} problem(s)`);
  process.exit(1);
}
console.log('render check OK: no raw markdown leaks, all internal links resolve');
