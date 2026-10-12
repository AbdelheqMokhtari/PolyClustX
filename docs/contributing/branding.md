# Brand assets and affiliations

**PolyClustX** is the display name. Use `polyclustx` for the distribution,
Python import and command, and `PolyClustX` for the GitHub repository name.

> A Unified Framework for Multi-View and Multimodal Representation Learning and
> Clustering from Feature Views to Attributed Graphs

## Shared light palette

`src/polyclustx/brand.py` defines the palette, description and project links.
The interface, launcher and Sphinx templates read this source. Use white and
near-white surfaces, indigo for actions, and violet/steel blue as secondary
accents. Keep dark colors for readable text, not large background panels.

The image files are packaged in `src/polyclustx/display/assets/` so installed
interfaces work without network access. Sphinx copies the same assets into its
static output; do not maintain separate copies.

## Project logo

`logo.png` is an original, AI-generated transparent raster mark produced using
the built-in image-generation tool on 12 October 2026. It contains only three circular
nodes joined by straight edges, replacing the earlier decorative X-shaped mark. Display it with
its original aspect ratio and alpha channel on a light surface.

The generation prompt was:

> Use case: logo-brand. Create a replacement project logo for PolyClustX, an academic clustering and representation-learning library. Extremely simple flat scientific mark: exactly three solid circular nodes at the vertices of a compact equilateral triangle, joined by three thin straight lines. No surrounding shape. Two nodes and lines in muted indigo #7478AE, one node in cool steel blue #557AA3. Balanced geometry, uniform strokes, generous clean negative space, readable at 24 pixels. Square canvas, single centered mark, modest padding. Genuinely transparent background with alpha. No text, no letters, no X, no curved swooshes, no orbit rings, no additional dots, no gradients, no shadows, no 3D, no texture, no watermark. Restrained and simple, suitable for scientific software.

## Institutional logos

The repository owner supplied `borelli.png` and `paris_cite.png` in the former
root `assets/` directory. They have been relocated to the packaged asset
directory without recoloring, cropping, redrawing or changing their file contents.
Display them separately, on white, with their full names and institution links.

- [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/fr)
- [Université Paris Cité — branding guidance](https://u-paris.fr/charte-graphique-et-outils/)

These identify research affiliations, not sponsorship or endorsement. They are
excluded from the software's MIT license. Their presence is not a grant of
permission for other uses; confirm any required institutional approval before
public release and follow the institutions' current guidelines.
