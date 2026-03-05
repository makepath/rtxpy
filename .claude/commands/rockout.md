# Rockout: End-to-End Issue-to-Implementation Workflow

Take the user's prompt describing an enhancement, bug, or suggestion and drive it
through all seven steps below. The prompt is: $ARGUMENTS

---

## Step 1 -- Create a GitHub Issue

1. Decide the issue type from the prompt:
   - **enhancement** -- new feature or improvement
   - **bug** -- something broken
   - **suggestion / proposal** -- idea that needs design discussion
2. Pick labels from the repo's existing set. Always include the type label
   (`enhancement`, `bug`). Add topical labels when they fit
   (e.g. `documentation`, `GPU CI`).
3. Draft the title and body. Write a clear problem statement, motivation, and
   proposed approach. For bugs, include reproduction steps and expected vs actual
   behavior.
4. **Run the body text through the `/humanizer` skill** before creating the issue
   to strip AI writing patterns.
5. Create the issue with `gh issue create` using the drafted title, body, and labels.
6. Capture the new issue number for later steps.

## Step 2 -- Create a Git Worktree

1. Create a new branch and worktree using the issue number:
   ```
   git worktree add .claude/worktrees/issue-<NUMBER> -b issue-<NUMBER>
   ```
2. Switch the working directory to the new worktree for all remaining steps.

## Step 3 -- Implement the Change

1. Read the relevant source files to understand the existing code.
2. Follow the project's architecture patterns:
   - **xarray accessor**: public API goes through `rtxpy/accessor.py` (`RTXAccessor`)
   - **Engine/viewer**: interactive features in `rtxpy/engine.py` and `rtxpy/viewer/` subsystems
   - **Analysis**: raster operations in `rtxpy/analysis/` with `prepare_mesh()` in `_common.py`
   - **Mesh utils**: geometry operations in `rtxpy/mesh.py`
   - **Data fetching**: `fetch_*` functions for remote data sources
3. When adding viewer features, use the key dispatch pattern:
   add entries to `KEY_BINDINGS` / `SHIFT_BINDINGS` / `SPECIAL_BINDINGS` in
   `rtxpy/viewer/keybindings.py`, then add thin action methods on `InteractiveViewer`.
4. When adding analysis functions, return standard xarray DataArrays.
5. Keep changes focused -- don't refactor surrounding code unnecessarily.

## Step 4 -- Add Test Coverage

1. Add or update tests in `rtxpy/tests/`.
2. Any temporary files must have unique names. Include the issue number in
   the filename (e.g. `tmp_42_result.zarr`) to avoid collisions with
   parallel test runs or other worktrees.
3. Cover:
   - Correctness against known values or reference implementations
   - Edge cases (NaN handling, empty input, single-cell rasters)
   - GPU and CPU code paths where applicable
4. Run the tests with `pytest rtxpy/tests/` to verify they pass before moving on.

## Step 5 -- Update Documentation

1. Check `docs/` for the relevant markdown file:
   - `docs/api-reference.md` -- method signatures and parameters
   - `docs/user-guide.md` -- task-oriented workflows
   - `docs/getting-started.md` -- installation and first steps
   - `docs/examples.md` -- annotated walkthroughs
2. Add or update the entry for any new public functions or viewer features.
3. If a new data fetcher or analysis function was created, add it to the
   appropriate section.

## Step 6 -- Create an Example

The project has an `examples/` directory with both `.py` scripts and `.ipynb` notebooks.

1. Choose the right format:
   - `.py` script for CLI-oriented workflows or `explore()` demos
   - `.ipynb` notebook for analysis workflows with inline visualization
2. Follow the established patterns from existing examples:
   - Fetch data from a bounding box using `fetch_*` functions
   - Show the analysis or placement pipeline
   - End with `explore()` or a rendered image
3. Use `matplotlib` for static plots, consistent with existing notebooks.
4. Keep the example self-contained (no external data dependencies beyond fetch calls).

**Skip this step** if the change is a pure bug fix with no new user-facing API.

## Step 7 -- Commit, Push, and Create PR

1. Stage and commit changes with a clear message referencing the issue number
   (e.g. `Add flood velocity function (#42)`).
2. Push the branch and create a pull request with `gh pr create`.
3. In the PR body, reference the issue (e.g. `Closes #42`) and summarize
   what was done.
4. **Run the PR body text through the `/humanizer` skill** before creating.

---

## General Rules

- Work entirely within the worktree created in Step 2.
- Commit progress after each major step with a clear commit message referencing
  the issue number.
- Run `/humanizer` on any text destined for GitHub (issue body, PR description)
  to remove AI writing artifacts.
- If any step is not applicable (e.g. no docs update needed for a typo fix),
  note why and skip it.
- At the end, print a summary of what was done and where the worktree lives.
