# Contributing to OMRAT

Thanks for wanting to improve OMRAT. This short document lists the
practical things to know before you send a pull request.

## Licensing of contributions

OMRAT is distributed under the **GNU General Public License, version 2
or (at your option) any later version** (GPL v2+). By opening a pull
request, you agree that:

1. You are the author of the contribution, or you have the right to
   submit it under GPL v2+ (for example because it is already
   compatibly-licensed open-source code).
2. Your contribution will be distributed as part of OMRAT under the
   same GPL v2+ terms.

This is a lightweight version of the *Developer Certificate of Origin*
(https://developercertificate.org/). No separate CLA is required.

If your contribution is substantial and your organisation needs a
formal assignment for their records, mention it in the pull request
and we'll sort out the paperwork before merging.

## Development setup

See `help/source/installation.rst` for user-facing install
instructions. For development:

```bash
git clone https://github.com/axelande/OMRAT.git
cd OMRAT

# Use the QGIS-shipped Python so QGIS bindings resolve.
# On Windows (OSGeo4W):
C:/OSGeo4W/apps/Python312/python.exe -m pip install -r requirements_dev.txt

# Symlink the repo into your QGIS plugin folder so QGIS can load it.
# Paths are in help/source/installation.rst.
```

## Running tests

Most tests run without a live QGIS instance:

```bash
# From the repo root:
C:/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis --noconftest tests/
```

The `--noconftest -p no:qgis` flags skip the test fixtures that need a
full QGIS environment. A small number of tests still require QGIS;
they're documented in `CLAUDE.md` under "Test file categories" and
run in CI via the docker-qgis job (`.github/workflows/python_app.yml`).

## Code style

- **No QGIS imports in `compute/` or the standalone geometry
  modules.** Those modules must remain usable outside QGIS (the test
  suite depends on it, and so does the headless profiler under
  `tests/diagnostics/`).
- **ASCII in files Windows cp1252 may read.** Use `->` instead of
  `→`, avoid unicode in docstrings unless there is a reason.
- **Cast depth values with `float()`.** They are stored as strings in
  the `.omrat` file.
- **Along-track distance must be projected onto leg direction** --
  never shortest Euclidean distance.

## Pull request checklist

- [ ] Tests added for the new / changed behaviour.
- [ ] Existing tests pass (`pytest tests/`).
- [ ] Docstrings / doc-source updated if user-visible behaviour
      changed.
- [ ] No new Sphinx warnings introduced (`sphinx -b html -n`).
- [ ] `.omrat` file-format compatibility preserved (adding new keys
      is fine; changing the meaning of existing keys needs a
      migration step in
      `omrat_utils/storage.py:_normalize_legacy_to_schema`).

## Filing issues

Please include:

- QGIS version and OS.
- OMRAT version (from `metadata.txt` or the Plugin Manager).
- A minimal reproducer -- the smallest `.omrat` file that triggers
  the issue.
- The OMRAT log tab output from **View -> Panels -> Log Messages
  Panel -> OMRAT**.

## Citing OMRAT

If you use OMRAT in a publication or report, please cite it using
the `CITATION.cff` file at the root of the repository. GitHub renders
a "Cite this repository" button from that file.
