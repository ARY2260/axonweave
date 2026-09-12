# Security Policy

:::DOC-NOTE
The canonical policy is maintained in the repository root at `SECURITY.md`; this page mirrors it in the web documentation.
:::

AxonWeave is an open-source scientific library that ships a small Python package,
an optional Rust/PyO3 native extension, and a static documentation website. This
page documents supported versions, how to report security issues, and the
security expectations we commit to.

## Supported Versions

AxonWeave is pre-1.0. Security fixes are released in the latest stable release
and backported only where the fix is trivial and the affected version is still
supported.

| Version | Supported |
| ------- | --------- |
| Latest stable release (0.1.x) | Yes |
| Previous stable releases | Security fixes only, on a best-effort basis |
| `main` branch | No; maintained for maintainers and contributors only |
| Nightly built docs | No |

For release channels, see the [distribution](distribution.md) page.

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.** Private disclosure
gives maintainers time to assess and fix the issue before it is disclosed.

Report vulnerabilities privately through GitHub's private advisory workflow:

    https://github.com/dhakalnirajan/axonweave/security/advisories/new

### What to include

To help us triage quickly, include:

- The affected component (Python package, Rust extension, documentation site).
- Version information (`pip show axonweave`, commit SHA, or wheels you used).
- A minimal, reproducible description of the issue — what you did, what you
  observed, and what you expected.
- Impact: what is at risk, and who could be affected.
- Any suggested remediation, if you have one.

### What happens next

- **Acknowledgment:** maintainers will acknowledge the report within 48 hours.
- **Triage:** a severity assessment and next steps within 5 business days.
- **Fix and release:** a fix is prepared and released in the next stable
  release, or a workaround is documented if the issue cannot be fixed.
- **Coordinated disclosure:** the issue is disclosed publicly (e.g., a GitHub
  Security Advisory) after a fix is available, unless the reporter requests
  otherwise.

If a report is declined as out of scope (below), we will respond with a
reasoned explanation.

## Security scope

### In scope

- The `axonweave` Python package: provisioning, substrate loading, graph
  assembly, model execution, checkpoints, and configuration parsing.
- The Rust/PyO3 extension (`axonweave._native`).
- The documentation website and its build/deploy pipeline
  (`.github/workflows/docs.yml`), including safe rendering of documentation
  content and the absence of client-side secrets (see `docs-site/SECURITY_HEADERS.md`).
- Dependency and supply-chain hygiene for the wheels and the docs site.

### Out of scope

- Calibration, accuracy, or biological-plausibility issues in the mathematical
  model. These are scientific questions, not security vulnerabilities; file
  them as issues or raise them through the scientific review process.
- License or provenance compliance of upstream data (e.g., the MaleCNS v1.0
  connectome). These are governed by the upstream data license; see
  `DATA_LICENSE.md` and `DATA_SOURCES.md`.
- Vulnerabilities in upstream dependencies. Report these to the upstream
  project; we track dependency updates through our dependency review in CI.
- Self-inflicted exposure, such as committed credentials or accidentally
  published private data (though reporting it is still appreciated).

## Non-negotiable security rules

These rules are enforced project-wide and in CI:

1. **Never commit secrets.** API keys, PyPI tokens, cloud credentials, private
   URLs, and private user datasets must never appear in the repository, CI logs,
   or built artifacts.
2. **No client-side secrets in the docs site.** Analytics identifiers, if any,
   must be configured through deployment environment variables and a
   privacy-preserving provider. The built site is public static content.
3. **The wheel stays independent of biological data.** Multi-gigabyte upstream
   data is provisioned and cached separately and is never bundled into the PyPI
   package.
4. **No generated build artifacts in version control.** Build output, `dist/`,
   caches, and local data directories remain ignored.
5. **Errors carry no sensitive data.** Error messages should never leak
   paths, keys, or user data beyond what is needed for an actionable
   `AXW###`-coded diagnostic.

## Responsible disclosure

We follow coordinated disclosure: we do not demand silence forever, but we ask
that disclosures are coordinated with maintainers so that fixes can ship before
or together with the public announcement. If you plan to publish research based
on a vulnerability, contact us first.