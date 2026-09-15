## Responsive behavior

All grid elements collapse to single-column on screens narrower than 960px. The sidebar becomes an off-canvas drawer toggled by the hamburger icon; the right-side TOC is hidden; code blocks become full-width. The cookie banner stacks vertically on small screens.

```css
/* Mobile-first: single column, sidebar drawer, no TOC, no resizer. */
@media (max-width: 960px) {
  .shell {
    grid-template-columns: 1fr !important;
    padding: 20px 16px 60px;
    gap: 0;
  }
  .sidebar {
    position: fixed; left: 0; top: 0; bottom: 0;
    width: 280px; z-index: 100; background: var(--surface);
    border-right: 1px solid var(--line);
    transform: translateX(-100%); transition: transform .25s ease;
    padding: 20px 18px; overflow-y: auto;
  }
  .sidebar.open { transform: translateX(0); }
  .sidebar-resizer { display: none; }
  .toc { display: none; }
  .content article { max-width: 100%; }
  .search-trigger { min-width: 0; padding: 8px 10px; }
  .search-trigger span, .search-trigger kbd { display: none; }
  .topnav { display: none; }
  footer { flex-direction: column; align-items: center; gap: 8px; text-align: center; }
}
@media (max-width: 600px) {
  .cookie { flex-direction: column; gap: 12px; text-align: center; }
  .cookie .cookie-actions { justify-content: center; }
  .page-nav { flex-direction: column; }
  .page-nav-cell { max-width: 100%; }
  .search-dialog { margin: 20px; }
}
```

## Cookie banner

The banner follows standard consent-management behavior: Accept all, Reject all, Manage preferences link, ability to close, ARIA compliance, preference persistence.

```tsx
function CookieConsent() {
  const [show, setShow] = useState(false);
  const [prefs, setPrefs] = useState(false);
  const cookieKey = 'axonweave-cookie-consent';

  useEffect(() => {
    const stored = localStorage.getItem(cookieKey);
    if (!stored) setShow(true);
  }, []);

  const consent = (value: string) => {
    localStorage.setItem(cookieKey, value);
    setShow(false);
    setPrefs(false);
  };

  if (!show) return null;
  return (
    <div className="cookie" role="dialog" aria-label="Cookie preferences">
      {!prefs ? (
        <>
          <div>
            <strong>Privacy & cookies</strong>
            <p>
              This site uses essential cookies only. Optional analytics, if
              enabled, will be disclosed in the Privacy Policy. You may accept
              all or reject non-essential cookies.
            </p>
          </div>
          <div className="cookie-actions">
            <button className="primary" onClick={() => consent('all')}>
              Accept all
            </button>
            <button className="secondary" onClick={() => consent('essential')}>
              Reject all
            </button>
            <button className="ghost" onClick={() => setPrefs(true)}>
              Manage preferences
            </button>
          </div>
        </>
      ) : (
        <>
          <div>
            <strong>Manage preferences</strong>
            <label className="cookie-pref">
              <input type="checkbox" disabled checked />
              Essential (always on)
            </label>
            <label className="cookie-pref">
              <input type="checkbox" />
              Analytics (coming soon)
            </label>
          </div>
          <div className="cookie-actions">
            <button className="primary" onClick={() => consent('all')}>
              Save and accept
            </button>
            <button className="ghost" onClick={() => setPrefs(false)}>
              Back
            </button>
          </div>
        </>
      )}
    </div>
  );
}
```

## CSS additions

```css
/* Cookie banner states */
.cookie { position: fixed; bottom: 0; left: 0; right: 0; z-index: 50;
  background: var(--surface); border-top: 1px solid var(--line);
  box-shadow: 0 -10px 30px rgba(0,0,0,.2); padding: 18px 24px;
  display: flex; gap: 24px; align-items: center; justify-content: space-between; }
.cookie p { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
.cookie-actions { display: flex; gap: 8px; flex-shrink: 0; }
.secondary { background: var(--surface-2); border: 1px solid var(--line); color: var(--text);
  border-radius: 5px; padding: 10px 15px; font-weight: 600; cursor: pointer; }
.secondary:hover { border-color: var(--accent); }
.ghost { background: none; border: none; color: var(--muted); padding: 10px 12px;
  cursor: pointer; font-size: 14px; }
.ghost:hover { color: var(--text); }
.cookie-pref { display: flex; gap: 8px; align-items: center; font-size: 14px;
  color: var(--text); margin: 8px 0; cursor: default; }
.cookie-pref input { accent-color: var(--accent); }

/* Responsive */
@media (max-width: 960px) {
  .shell { grid-template-columns: 1fr !important; padding: 20px 16px 60px; gap: 0; }
  .sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 280px; z-index: 100;
    background: var(--surface); border-right: 1px solid var(--line);
    transform: translateX(-100%); transition: transform .25s ease;
    padding: 20px 18px; overflow-y: auto; }
  .sidebar.open { transform: translateX(0); }
  .sidebar-resizer { display: none; }
  .toc { display: none; }
  .content article { max-width: 100%; }
  .search-trigger { min-width: 0; padding: 8px 10px; }
  .search-trigger span, .search-trigger kbd { display: none; }
  .topnav { display: none; }
  footer { flex-direction: column; align-items: center; gap: 8px; text-align: center; }
  .code-block-actions { gap: 6px; }
}
@media (max-width: 600px) {
  .cookie { flex-direction: column; gap: 12px; text-align: center; }
  .cookie .cookie-actions { justify-content: center; flex-wrap: wrap; }
  .page-nav { flex-direction: column; }
  .page-nav-cell { max-width: 100%; }
  .search-dialog { margin: 12px; }
}
```
