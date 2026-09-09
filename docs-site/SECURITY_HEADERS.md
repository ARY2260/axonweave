# Documentation Deployment Security

GitHub Pages serves the site over HTTPS. The application contains no API secrets and does not require client-side credentials.

For a custom reverse proxy/CDN, configure:

- `Strict-Transport-Security`
- `Content-Security-Policy` allowing only the documentation origin, Google Fonts if retained, and the selected analytics provider if enabled
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` with only required browser features

Analytics is opt-in at deployment level through `VITE_ANALYTICS_DOMAIN`; do not place secret analytics tokens in source code.
