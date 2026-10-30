# Third-party sources

The vendored `eliciting-secret-knowledge` and `jacobian-lens` directories are
unmodified source snapshots migrated from `secret_keeping_old`. Their upstream
README, license, and package metadata remain in each directory. Framework code
imports these packages only where compatibility requires it; no training or
fitting entry point is called by `tools/run_net.py`.

Upstreams:

- <https://github.com/cywinski/eliciting-secret-knowledge>
- <https://github.com/Butanium/jacobian-lens>

