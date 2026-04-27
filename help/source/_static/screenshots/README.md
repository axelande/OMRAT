# Screenshot capture for the OMRAT documentation

The rewritten user-facing docs reference a fixed set of screenshot
filenames in this folder.  There are two ways to populate them:

1. **Automatic** -- run the helper script
   [`tools/capture_screenshots.py`](../../../../tools/capture_screenshots.py)
   inside QGIS once.  It walks the plugin's tabs + settings dialogs,
   grabs each to PNG with the exact filenames below, and saves them
   here.
2. **Manual** -- snip by hand against the checklist below.  Filenames
   must match exactly.

Any slot without a file shows up as a broken image in the HTML build
and Sphinx emits a warning for it.  The build still succeeds.

## Running the automatic capture

The script needs the real Qt event loop (widget `.grab()` doesn't
render off-screen consistently), so it runs inside a QGIS instance.

**Interactive (easiest):**

1. Open QGIS.
2. Enable OMRAT in the Plugin Manager.
3. **Plugins -> Python Console**.
4. In the console type:

   ```python
   exec(open(r'e:/dev_e/program/OMRAT/tools/capture_screenshots.py').read())
   ```

   (Adjust the path to your checkout.)  The console prints the list
   of PNGs as they're written.

**Batch (OSGeo4W shell on Windows):**

```
python-qgis tools/capture_screenshots.py
```

**Batch (Linux):**

```
qgis --nologo --code tools/capture_screenshots.py
```

The script starts with dry shots (no real Run Model + no Drift
Analysis, since both are slow).  If you want the "after run" screens
to show real computed numbers instead of synthetic placeholders, edit
the last line of `tools/capture_screenshots.py`:

```python
main(run_model=True, drift_analysis=True)
```

and re-run.  A full run can take several minutes on proj.omrat.

## Filename checklist

Filenames below are exactly what the docs reference.  The automatic
script writes all of them except `ui_drift_corridor.png` when
`drift_analysis=False` and a few "-filled" variants when
`run_model=False` (synthetic numbers are substituted instead).

### Landing + quickstart

- [ ] `ui_dock_overview.png`
- [ ] `ui_toolbar.png`
- [ ] `ui_loaded_example.png`
- [ ] `ui_run_model.png`
- [ ] `ui_task_progress.png`
- [ ] `ui_results_filled.png`
- [ ] `ui_result_layers.png`  *(needs `run_model=True`)*

### User guide (tab-by-tab)

- [ ] `ui_dock_tabs_annotated.png`  *(the script writes the dock;
  add callouts in an image editor afterwards)*
- [ ] `ui_tab_route.png`
- [ ] `ui_tab_traffic.png`
- [ ] `ui_tab_depths.png`
- [ ] `ui_tab_objects.png`
- [ ] `ui_tab_distributions.png`  *(same content as Route tab; the
  distribution controls live inside Route_tab in the current UI)*
- [ ] `ui_tab_drift_analysis.png`
- [ ] `ui_tab_results.png`

### Drift Analysis output

- [ ] `ui_drift_corridor.png`  *(needs `drift_analysis=True`)*

### Settings dialogs

- [ ] `ui_settings_drift.png`
- [ ] `ui_settings_causation.png`
- [ ] `ui_settings_ship_categories.png`
- [ ] `ui_settings_ais.png`

## Capture environment (manual or machine)

- **QGIS version**: 3.30 LTS or newer.
- **OMRAT**: installed via the Plugin Manager, with the
  `tests/example_data/proj.omrat` project loaded (the script does
  this for you).
- **Theme**: QGIS light theme preferred.  Dark mode works but be
  consistent across the set.
- **Resolution**: 1920 x 1080 minimum.  Save as PNG (lossless).
- **Redaction**: the example project contains no sensitive data.  If
  you re-capture from a real project, review for port / client names.

## Tips

- The script grabs the dock widget at 520 x 820 to get full tab
  content without scrollbars.  Adjust `DOCK_WIDTH` / `DOCK_HEIGHT` in
  the script if your project needs more room.
- `ui_dock_tabs_annotated.png` is a base shot you can open in an
  image editor and decorate with callouts (menu bar, tab row, status
  bar).  The script doesn't try to do the annotation itself.
- `ui_toolbar.png` grabs whichever QToolBar contains an "OMRAT"
  action.  If your install labels it differently, edit
  `grab_toolbar` in the script.
- QGIS has a "Decorations -> Copyright label" that some analysts
  leave on -- turn it off before capturing so the canvas shots are
  clean.

## After you drop images in

Rebuild the HTML docs:

```
cd help
python -m sphinx -b html --keep-going source build/html
```

The landing page, quickstart, and user_guide will render the real
screenshots in place of the current alt-text placeholders.  Sphinx
still warns for any slot you haven't filled, but the build succeeds.
