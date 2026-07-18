# Security policy

## Supported versions

Security fixes are provided for the latest published major/minor release. Users
should run a supported Python version and current compatible dependencies.

## Reporting a vulnerability

Please use GitHub's private security-advisory reporting for
`ToucanDB/ToucanDB`. Do not open a public issue with an exploit, encryption key,
or sensitive database. Include affected versions, impact, a minimal
reproduction, and any proposed mitigation.

## Security boundary

- ToucanDB is an embedded library, not an authentication or authorization
  service. The host application must authorize every collection, namespace,
  metadata filter, and returned source.
- One process owns a database directory. The lock prevents accidental second
  owners; filesystem permissions must prevent untrusted users from replacing
  database files.
- Optional Fernet encryption protects vector and metadata payloads. It does not
  encrypt schemas, vector IDs, SQLite structure, hashed keys, lock metadata, or
  FAISS index snapshots. Use platform/full-disk protection for whole-directory
  secrecy.
- Keys are derived with scrypt and a random per-database salt. Keys are never
  stored in the database or backup and cannot be recovered.
- A key must come from a keychain, secret manager, or host-controlled
  environment. Do not hard-code it or log it.
- Backups contain the same sensitive data and salt as the source. Protect them
  with equivalent filesystem controls and retain the key separately.
- Retrieved documents are untrusted prompt input. RAG prompting reduces but
  cannot eliminate prompt-injection risk.

## Dependency and release practices

CI tests every supported Python version, builds the distribution, checks
metadata, and publishes with PyPI trusted publishing from the release workflow.
Release changes should review dependency advisories and avoid unnecessary
runtime packages.
