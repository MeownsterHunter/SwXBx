## What changed

Describe the behavior and the contract it affects.

## Verification

List the commands and observable results that verify the change.

## Checklist

- [ ] Public interfaces and ownership rules are documented.
- [ ] Tests cover success, failure, and boundary behavior.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes.
- [ ] `cargo test --workspace --all-features` passes.
- [ ] The change contains no private paths, credentials, captured program data, or proprietary material.
