---
version: alpha
name: AxonWeave Docs
description: Design tokens for the AxonWeave documentation site. Dark theme is default; light theme is the inverse. Typography uses standard system sans-serif/monospace stacks (like PyTorch and TensorFlow docs) instead of a single custom webfont. Tokens map 1:1 to CSS custom properties in src/style.css.
colors:
  primary: "#b7e2ef"
  bg: "#121212"
  surface: "#1a1c1e"
  surface-2: "#242628"
  text: "#e8eaed"
  muted: "#9aa4ad"
  line: "#33383d"
  accent: "#8dc4d8"
  accent-strong: "#b7e2ef"
  brand: "#ee4c2c"
  code: "#18181b"
  code-border: "#2e3135"
  on-primary: "#121212"
  light-bg: "#f8f9fa"
  light-surface: "#ffffff"
  light-surface-2: "#eef1f3"
  light-text: "#202124"
  light-muted: "#5f6b76"
  light-line: "#dadce0"
  light-accent: "#17677e"
  light-accent-strong: "#0f5266"
  light-code: "#18181b"
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

AxonWeave Documentation Frontend Design System. A restrained, high-contrast documentation UI in the PyTorch/TensorFlow technical-docs tradition: deep charcoal canvas (#121212 dark / #f8f9fa light), a single steel-blue accent for links, and a coral-red brand color (#ee4c2c) reserved for active states. Standard system sans-serif for reading and system monospace for code. Documentation first — nothing decorative competes with text.

## Colors

Dark theme (default) is normative; light theme tokens are prefixed `light-` and map to the same CSS custom properties under `:root[data-theme=light]`.

- **bg / surface / surface-2:** layered charcoal surfaces for page, cards and hover states.
- **text / muted:** primary reading text and secondary metadata.
- **line:** hairline borders only — never colored card borders.
- **accent / accent-strong:** links and small highlights. Not used for gradients.
- **brand:** coral red-orange (#ee4c2c), reserved exclusively for active states and focal points (TOC active marker).
- **code / code-border:** code containers keep a dedicated dark IDE aesthetic (#18181b) in BOTH themes — code blocks do not change with the page theme.

## Typography

Standard system font stacks, following the convention of PyTorch and TensorFlow documentation: system sans-serif (`system-ui`) for UI/body/headings and system monospace (`ui-monospace`) for code. No webfont download, no FOUC, native rendering per OS. Headings use tight negative tracking. `label-caps` is for sidebar group labels and TOC headers only.

## Layout

Hairline borders use `line` / `light-line` (decorative, excluded from WCAG text-contrast rules). Fixed top bar carries brand, structural tabs (Learn, API, Tutorials, GitHub), a visible search trigger with Shift+/ shortcut, and a stable/nightly version selector. Desktop: top navigation, left documentation sidebar (user-resizable via a drag handle, persisted to localStorage), centered reading column (max 760px), right-side "On this page" TOC with scroll-position tracking. Mobile: collapsible navigation drawer, single reading column, TOC hidden.

## Callouts

Advisory boxes are flat, with a 3px solid left accent bar: blue for notes, amber for constraints/warnings, green for tips. Solid fills, no shadows, no glassmorphism.

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
