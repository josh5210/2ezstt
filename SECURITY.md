# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly (if contact info available) or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Security Considerations

### API Keys
- Never commit API keys or tokens
- Use environment variables for all secrets
- The `.env` file is gitignored by default

### Network Security
- The WebSocket server binds to `127.0.0.1` by default (local only)
- Change `EZSTT_HOST` to `0.0.0.0` only if you understand the implications
- Consider using `EZSTT_AUTH_TOKEN` for authentication in production

### Audio Data
- Audio is processed in memory and not persisted by default
- No audio is sent to external services unless explicitly configured (e.g., OpenAI backend)

## Response Timeline

We aim to:
- Acknowledge reports within 48 hours
- Provide an initial assessment within 7 days
- Release patches for critical vulnerabilities ASAP
