# Version

Version No.

```text
  v{major}.{minor}.{stage}.{revision}
```

| stage | No. | description | e.g |
| --- | --- | --- | --- |
| Alpha | 0.5 | smoke test, estimate gains  | v0.0.5.0 |
| Beta | 0.7 | integration test  | v0.0.7.2 |
| RC1 | 0.8 | stress test  | v0.0.8.1 |
| RC2 | 0.9 |  AB-test, evaluate gains  | v0.0.9.0 |
| Release | 1.0 |  production  | v0.1.0.0 |

# Release Version

Make sure all PRs under milestone `v0.3.2` are closed, then close the milestone.
Using below command to generate relase note.

`python tools/release_notes.py -c didi delta v0.3.2`
