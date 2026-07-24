# Security Policy

SimWorld is a self-contained research simulation. It ships **no secrets**, makes **no
outbound network calls** in its default (`make smoke`) configuration, and operates
entirely on synthetic, self-generated data — there is no real user or personal data
anywhere in the pipeline. The realistic attack surface is small, but we take reports
seriously.

## Supported versions

The `main` branch is the only supported version. Fixes land there.

## Reporting a vulnerability

Please **do not** open a public issue for a security-sensitive report. Instead, use
GitHub's [private vulnerability reporting](https://github.com/alyssamatsuzaki/simworld/security/advisories/new)
("Report a vulnerability" under the repository's **Security** tab). Include:

- a description of the issue and its impact,
- the steps or a minimal proof of concept to reproduce it, and
- any suggested remediation.

You can expect an acknowledgement within a few days. Once a fix is available we will
credit you in the release notes unless you prefer to remain anonymous.

## Scope notes

- **Optional networked backends** (Weights & Biases, a remote Ray cluster) are the only
  components that read credentials, and only when you explicitly opt in via `.env`.
  Keep your `.env` out of version control — it is git-ignored by default.
- **Dependencies** are pinned via `uv.lock`; run `uv lock` and review advisories before
  bumping. Reports about a vulnerable transitive dependency are welcome.
