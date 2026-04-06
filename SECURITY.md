# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please do NOT report security vulnerabilities through public GitHub issues.

Instead, email: gwthm.in@gmail.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- Acknowledgment: within 48 hours
- Initial assessment: within 1 week
- Fix timeline: depends on severity, typically within 2 weeks for critical issues

## Scope

The following are in scope:
- SQL injection in the SQLite store
- Path traversal in file operations
- Prompt injection via stored content
- Information leakage between conversations
- Authentication/authorization bypass in adapters
