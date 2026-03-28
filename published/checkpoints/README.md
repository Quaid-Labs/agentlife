# Published Checkpoints

Put frozen public-supporting benchmark artifacts here by release tag.

During prep, stage the pending public-supporting files in
`release-candidate/`, then copy them into the final release-tag directory once
the public claims are locked.

Suggested layout:

```text
published/checkpoints/
  release-candidate/
    README.md
    scores/
    token-usage/
    selected-eval-results/
  2026-03-release/
    README.md
    scores/
    token-usage/
    selected-eval-results/
```

Only copy artifacts that support released claims. Raw working-state benchmark
directories should stay outside this tree.
