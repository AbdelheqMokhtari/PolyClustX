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
the built-in image-generation tool on 12 October 2026. It suggests a handwritten
mathematical X with connected nodes and grouped observations. Display it with
its original aspect ratio and alpha channel on a light surface.

The generation prompt was:

> Use case: logo-brand. Asset type: transparent square logo mark for PolyClustX, an academic Python framework for multi-view and multimodal representation learning and clustering, from feature views to attributed graphs. Create one original minimal scientific symbol: a graceful mathematical handwritten-style X formed by two clean curved strokes, with a small purposeful arrangement of connected circular nodes and grouped dots integrated into the strokes, suggesting multiple feature views converging into graph clusters. Elegant geometry, balanced negative space, precise flat vector-like edges, legible at small sidebar/favicon scale. No literal words, no letters other than the abstract X-like form, no equations, no extra ornaments. Palette exclusively Borelli indigo #4E4E92 and lighter #7478AE, muted violet #634E87, cool steel blue #557AA3; prefer the lighter indigo and steel blue, with sparse primary indigo accents. Airy, light, scientific, not dark or heavy. Centered single mark taking most of the canvas with modest transparent padding. Genuinely transparent background with alpha, no white rectangle, no checkerboard drawn into the artwork. No gradients, no shadows, no glow, no 3D, no mockup, no watermark. Do not incorporate institutional logos.

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
