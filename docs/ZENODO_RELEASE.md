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
5. Confirm that the source archive excludes manuscript files:

   ```bash
   git archive --format=tar HEAD | tar -tf - | grep '^paper/' && exit 1 || true
   ```

6. Create an annotated version tag. Do not reuse or
   move an existing tag after Zenodo has archived it.
7. Push the tag and create a GitHub Release from that exact tag.
8. Wait for Zenodo to archive the release.
9. Verify the creator, ORCID, title, version, license, and files on Zenodo.
10. Record both the version DOI and concept DOI in the repository.

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
licensed under Apache-2.0. `paper/` is marked `export-ignore` and must not
appear in a GitHub/Zenodo software archive. The manuscript is distributed
separately through bioRxiv under the license shown on its record. Bundled
third-party images retain their upstream terms.
