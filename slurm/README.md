# Slurm templates

Scheduler templates are intentionally machine-neutral: they must not contain a
cluster account, username, private environment path, or output root. Site
specific values should be supplied through environment variables or an
untracked local wrapper.

The historical scripts are not copied here because their hard-coded NCHC paths
are provenance artifacts rather than portable launch interfaces.
