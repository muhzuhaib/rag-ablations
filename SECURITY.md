# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

Use GitHub's private reporting form:
[Report a vulnerability](https://github.com/muhzuhaib/rag-ablations/security/advisories/new). It
opens a thread visible only to you and the maintainer.

You can expect an acknowledgement within a week. If a report is valid, the fix and an advisory are
published together, and you are credited unless you would rather not be.

## Supported versions

This project is pre-1.0 and only the latest release on PyPI is supported. Fixes land on `main` and go
out in the next release rather than being backported.

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## What is in scope

- **The published package.** Anything in [`rag-ablations` on
  PyPI](https://pypi.org/project/rag-ablations/) that executes something it should not, or reads or
  writes outside the paths it documents.
- **Corpus loading.** BEIR archives are downloaded and unpacked. Anything that lets a crafted archive
  write outside the extraction directory is a vulnerability, not a bug.
- **The retrieval service.** The HTTP API answers queries against a local index. Anything that lets a
  query reach the filesystem, the process environment, or a different index is in scope.

## What is not in scope

- **The corpora themselves.** The datasets are third-party public research corpora, fetched from
  their published locations. Their contents are not this project's to vouch for.
- **The demonstration container.** The shipped image is built to run the benchmark and answer
  queries. It is not a hardened deployment and does not claim to be.
- **A benchmark figure you disagree with.** That is an issue, and a welcome one, but it is not a
  security report. Open it publicly with the measurement you got.
