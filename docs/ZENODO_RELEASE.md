# Zenodo release procedure

This repository uses Zenodo's GitHub integration to archive immutable
software releases and assign a DOI.

## One-time setup

1. Sign in to Zenodo using the GitHub account that owns the repository.
2. Open the Zenodo GitHub settings page.
3. Enable `jkoba0512/myoarm-forward-state-estimation`.
4. Confirm that Zenodo can see the repository.

Do not archive the existing `submission-ready/v1` tag. It predates the
R3 rerun and does not represent the current manuscript.

## Release checklist

1. Confirm that the bioRxiv DOI, if available, is recorded in
   `CITATION.cff`, `.zenodo.json`, `README.md`, and the manuscript.
2. Confirm that `README.md` describes the current manuscript and commands.
3. Run:

   ```bash
   uv sync --locked
   uv run pytest -q
   (cd paper && ./build.sh --clean)
   git diff --check
   ```

4. Confirm that the worktree is clean and the release commit is pushed.
5. Create an annotated version tag, initially `v0.1.0`. Do not reuse or
   move an existing tag after Zenodo has archived it.
6. Push the tag and create a GitHub Release from that exact tag.
7. Wait for Zenodo to archive the release.
8. Verify the creator, ORCID, title, version, license, and files on Zenodo.
9. Record both the version DOI and concept DOI in the repository.

## DOI policy

- Cite the **version DOI** when referring to the exact code used by the
  manuscript.
- Use the **concept DOI** in general project documentation when a link to
  the latest archived release is intended.
- The bioRxiv DOI identifies the manuscript. The Zenodo DOI identifies the
  archived software release. They should be linked as related research
  outputs, not treated as substitutes.

## Licensing

Repository software, configurations, and original project documentation are
licensed under Apache-2.0. `paper/main.tex` and `paper/main.pdf` are separate
manuscript works and follow the license shown on the bioRxiv record. Bundled
Springer Nature template files and attributed third-party images retain their
upstream terms.
