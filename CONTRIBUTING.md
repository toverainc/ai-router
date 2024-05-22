# Contributing to AI Router

## Found a bug?

If you think you found a bug, please check [Issues](https://github.com/toverainc/ai-router/issues) first, to see if it's not already reported. If the bug is already reported, avoid adding comments that add no additional information like "I am experiencing the same bug". Instead, add a thumbs up reaction to the main comment.

When opening a new issue, please use a clear title and description. Include a minimal config needed to reproduce the bug, and extra details about the backend if needed (e.g. vLLM v0.4.2 with mistralai/Mistral-7B-Instruct-v0.1). Also make sure not to include valid API keys!

## Code changes

Open a Github PR with your changes. If your changes fix an existing issue, [link the PR to the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

### Code formatting

Make sure your code is properly formatted by running `cargo fmt`. If your code is not properly formatted, CI will fail.

### Tests

Ideally you should add a test for the problem you are fixing, so that the problem will not reappear in the future.
Also make sure your change does not break any of the existing tests by running `cargo test`. CI will also run the tests, and fail if any of the tests fail.

### Clippy

While some people consider Clippy to be a nuisance and slows down development, we feel it helps to write better and more consistent Rust code. Therefore, we enable Clippy in CI. Please run `cargo clippy --all-features --all-targets --no-deps -- -Dclippy::pedantic` and make sure it succeeds.

### Commit message

Briefly describe what your commit does in the commit subject. If needed, elaborate in the commit message body. When in doubt, have a look at `git log` for examples.

### Bumping dependencies

When bumping direct dependencies (the ones defined in Cargo.toml), limit the bump to a single crate per commit, unless there is a need to bump other dependencies due to breakage. If you bump say 10 different crates in the same commit, and later a bug is found, it becomes difficult to bisect and revert which crate introduced the breakage.
