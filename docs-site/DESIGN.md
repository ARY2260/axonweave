---
version: alpha
name: AxonWeave Docs
description: Design tokens for the AxonWeave documentation site. Dark theme is default; light theme is the inverse. Typography uses standard system sans-serif/monospace stacks (like PyTorch and TensorFlow docs) instead of a single custom webfont. Tokens map 1:1 to CSS custom properties in src/style.css.
colors:
  primary: "#b7e2ef"
  bg: "#10151b"
  surface: "#151c23"
  surface-2: "#1b242d"
  text: "#e7edf2"
  muted: "#aeb8c2"
  line: "#2a343e"
  accent: "#8dc4d8"
  accent-strong: "#b7e2ef"
  code: "#0b1015"
  on-primary: "#10151b"
  light-bg: "#f6f7f8"
  light-surface: "#ffffff"
  light-surface-2: "#eef1f3"
  light-text: "#172027"
  light-muted: "#55616b"
  light-line: "#d9dfe4"
  light-accent: "#17677e"
  light-accent-strong: "#0f5266"
  light-code: "#f0f3f5"
typography:
  h1:
    fontFamily: system-ui
    fontSize: 44px
    fontWeight: 700
    lineHeight: "1.1"
    letterSpacing: "-0.04em"
  h2:
    fontFamily: system-ui
    fontSize: 27px
    fontWeight: 700
    lineHeight: "1.2"
    letterSpacing: "-0.025em"
  h3:
    fontFamily: system-ui
    fontSize: 20px
    fontWeight: 600
  body-md:
    fontFamily: system-ui
    fontSize: 16px
    lineHeight: "1.65"
  body-sm:
    fontFamily: system-ui
    fontSize: 14px
  label-caps:
    fontFamily: system-ui
    fontSize: 11px
    fontWeight: 700
    letterSpacing: "0.1em"
  code:
    fontFamily: ui-monospace
    fontSize: 13px
rounded:
  sm: 3px
  md: 5px
  lg: 6px
spacing:
  xs: 5px
  sm: 10px
  md: 18px
  lg: 28px
  xl: 48px
  gutter: 28px
components:
  button-primary:
    backgroundColor: "{colors.text}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.md}"
    padding: 10px 15px
  button-primary-hover:
    backgroundColor: "{colors.accent-strong}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.md}"
    padding: 10px 15px
  link:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.accent-strong}"
  link-hover:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.accent}"
  sidebar-active-item:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: 6px 10px
  topbar:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
  table-row:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.text}"
    padding: 9px 10px
  code-block:
    backgroundColor: "{colors.code}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: 18px
  copy-button:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: 5px 9px
  blockquote-note:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.muted}"
    padding: 15px 18px
  search-dialog:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
  cookie-banner:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
  light-theme:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-text}"
  light-theme-surface:
    backgroundColor: "{colors.light-surface}"
    textColor: "{colors.light-text}"
  light-theme-hover:
    backgroundColor: "{colors.light-surface-2}"
    textColor: "{colors.light-text}"
  light-theme-code:
    backgroundColor: "{colors.light-code}"
    textColor: "{colors.light-text}"
    rounded: "{rounded.lg}"
    padding: 18px
  light-theme-link:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-accent-strong}"
  light-theme-link-hover:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-accent}"
  light-theme-muted-text:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-muted}"
  light-theme-topbar:
    backgroundColor: "{colors.light-surface}"
    textColor: "{colors.light-text}"
  light-theme-table-row:
    backgroundColor: "{colors.light-bg}"
    textColor: "{colors.light-text}"
    padding: 9px 10px
---

## Overview

AxonWeave Documentation Frontend Design System. A restrained, high-contrast documentation UI: deep ink backgrounds (or clean paper in light mode), a single cool steel-blue accent, standard system sans-serif for reading and system monospace for code (matching the PyTorch/TensorFlow docs convention). Documentation first — nothing decorative competes with text.

## Colors

Dark theme (default) is normative; light theme tokens are prefixed `light-` and map to the same CSS custom properties under `:root[data-theme=light]`.

- **bg / surface / surface-2:** layered ink surfaces for page, cards and hover states.
- **text / muted:** primary reading text and secondary metadata.
- **line:** hairline borders only — never colored card borders.
- **accent / accent-strong:** links and small highlights. Not used for gradients.
- **code:** slightly darker than `surface` so code blocks read as inset.
- **on-primary:** text on primary buttons (primary buttons invert `text`/`bg`).

## Typography

Standard system font stacks, following the convention of PyTorch and TensorFlow documentation: system sans-serif (`system-ui`) for UI/body/headings and system monospace (`ui-monospace`) for code. No webfont download, no FOUC, native rendering per OS. Headings use tight negative tracking. `label-caps` is for sidebar group labels and TOC headers only.

## Layout

Hairline borders use `line` / `light-line` (decorative, excluded from WCAG text-contrast rules). Desktop: fixed top navigation, left documentation sidebar, centered reading column (max 760px), right-side TOC when space permits. Mobile: collapsible navigation drawer, single reading column, TOC hidden.

## Components

Required reusable components include navigation, sidebar, breadcrumbs, heading anchors, code block with copy action, admonition/note, table, source citation, search trigger, theme switcher, footer, cookie banner and 404 state.

## Do's and Don'ts

Hard constraints for every future frontend change:

- NO purple-to-blue gradients.
- NO gradient text in main headings.
- NO emojis in headings or titles.
- DO NOT use Inter as the default everywhere.
- NO glassmorphism cards or frosted-glass effects.
- NO colored borders on cards.
- NO row of exactly three icon boxes.
- NO cursor-following light beams.
- NO fade-in-on-scroll animations.
- NO grain overlay over gradients.
- Do not ship untouched default shadcn or UI-library components.
- Maintain high text contrast, especially in dark mode.
- No generic AI buzzwords in documentation copy.
- Do not use serif italic accents as decoration.
- Use the token spacing scale above, consistently.

### Accessibility

Use semantic landmarks, visible focus states, keyboard navigation, skip links, accessible names, sufficient contrast and responsive text sizing.

### Motion

Prefer no motion. If needed, use short state transitions for controls only. Never animate content into view on scroll.

### CTA

The landing page has exactly one primary CTA: **Install AxonWeave**. Secondary navigation actions must not visually compete with it.
