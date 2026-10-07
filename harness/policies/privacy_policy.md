# Privacy policy

A harnessed project keeps private material out of generic templates and
public commits.

High-risk material includes credentials, unpublished raw data, private
local paths, student work, reviewer names, private manuscripts, account
exports, double-blind identifying material, and AI-conversation logs.

Default handling:

- keep restricted material outside the repository;
- use local-only directories such as `data_local/`;
- scan commits before publication (`check_forbidden_phrases.py`,
  `check_path_markers.py`);
- replace private paths with stable project-relative paths;
- store no personal reviewer identities in the memory journal;
- report any possible exposure as a blocking risk.
