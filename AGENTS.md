# Project constraints

- Commit and push to GitHub after each crucial, verified change. Keep large
  downloaded artifacts and generated checkpoints out of Git.
- Stop after completing each phase; wait for the user before starting the next phase.

- Budget is zero: use local compute and free public downloads only. Do not use paid
  APIs, cloud GPUs, billable hosting, or services requiring payment.
- Keep project environments, downloaded runtimes, caches, temporary files, datasets,
  weights, checkpoints, logs, and reports on D:. Do not use C: for project storage.
- In PowerShell, dot-source `scripts/environment.ps1` before running project tools.
  This sets process-scoped paths; it does not change settings for other projects.
- Existing system tools may reside on C:. Do not move or delete shared installations
  or shared caches without checking ownership and scope.
- If the 4 GiB GPU cannot run a configuration, reduce resource requirements or
  document the limitation. Cloud spending is not a fallback.
