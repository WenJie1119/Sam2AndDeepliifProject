# Parallel development with Git worktrees

The repository uses one integration worktree and four task worktrees:

| Directory | Branch | Responsibility |
| --- | --- | --- |
| `CD34MicrovascularRecognition` | `main` | Integration and release testing |
| `CD34-pipeline-next` | `dev/pipeline-next` | Prompt, stitching, and pipeline changes |
| `CD34-ablation` | `exp/ablation` | Experiment matrices and evaluation |
| `CD34-sam3` | `feat/sam3-backend` | SAM3 backend and SAM2/SAM3 comparison |
| `CD34-dataset` | `data/dataset-v1` | Dataset manifests, conversion, and QC |

## Shared data

Ignored data and model files are not copied into new worktrees. Use absolute
paths rooted at:

```text
/local1/yangwenjie/CD34MicrovascularRecognition/data
```

Local per-worktree configuration files should be named `config/local*.json`;
these files are ignored by Git. Every concurrent run must use a different
output directory below:

```text
/local1/yangwenjie/CD34MicrovascularRecognition/debug_output/worktrees
```

## Synchronization

Before starting a new task in a feature worktree:

```bash
git status
git rebase main
```

If a branch is shared or already published, merge `main` instead of rebasing:

```bash
git merge main
```

Run the relevant tests and commit inside the feature worktree. Integrate from
the main worktree:

```bash
git switch main
git merge --no-ff <feature-branch>
python -m pytest -q
```

Avoid using the shared stash for routine context switching. Prefer a local WIP
commit on the task branch. Do not run concurrent jobs against the same output
or cache directory.

## Integration ownership

Feature branches should prefer adding isolated modules. Changes to the main
entry point, dependency declarations, default configuration, and top-level
README are coordinated in `main` after the feature module is tested.

Large WSI files, extracted image tiles, annotations, model weights, and run
outputs remain outside Git. Dataset branches commit scripts, manifests,
schemas, split definitions, and small synthetic test fixtures only.
