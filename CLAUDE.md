# Python Environment

Always use the conda environment `torch` when running Python.
Use `conda run -n torch python` instead of `python` for all Python commands.

Examples:
- Run a script: `conda run -n torch python script.py`
- Run a module: `conda run -n torch python -m pytest`
- Install packages: `conda run -n torch pip install <package>`

# Git Commit Rules

After every change (feature, fix, refactor, etc.), automatically run `git add -A` and `git commit` to stage ALL uncommitted changes (including untracked files), not just the files you modified.
Do NOT ask for confirmation before committing. Do NOT run `git push`.

### Commit Message Format
Follow the Conventional Commits standard:

    <type>(<scope>): <short summary>

    [optional body: explain WHY, not WHAT]

### Types
- `feat` – new feature
- `fix` – bug fix
- `refactor` – code restructure without behavior change
- `perf` – performance improvement
- `test` – adding or updating tests
- `docs` – documentation only
- `chore` – build process, dependencies, config

### Rules
- Summary line: max 72 characters, lowercase, no period at end
- Use imperative mood: "add user auth" not "added user auth"
- Scope is optional but encouraged: `fix(dataloader): handle empty batch`
- If the change is non-trivial, add a body explaining the reasoning
- Never use vague messages like "update", "fix bug", "changes", or "WIP"

### Good Examples
- `feat(model): add early stopping based on validation loss`
- `fix(preprocessing): handle missing values in age column`
- `refactor(trainer): extract epoch loop into separate method`
- `perf(dataloader): cache tokenized inputs to reduce CPU overhead`
- `chore: update torch dependency to 2.2.0`