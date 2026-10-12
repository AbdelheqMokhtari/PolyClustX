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
the built-in image-generation tool on 12 October 2026. It depicts interwoven
indigo and steel-blue bands suggesting connected data views. Display it with
its original aspect ratio and alpha channel on a light surface.

The generation prompt was:

> Use case: logo-brand. Create an original scientific software symbol for PolyClustX, multi-view representation learning and clustering. Completely new design: a compact elegant abstract woven lattice made of two intersecting pairs of gently curved bands, suggesting distinct data views coming together into one representation. Balanced mathematical geometry and clear negative space, like a refined scientific journal symbol. Flat vector-like rendering in muted indigo #7478AE, primary indigo #4E4E92 and cool steel blue #557AA3, with no gradients. Simple, distinctive silhouette; few shapes, medium-weight strokes, readable at 32 pixels. No text, letters, wordmark, triangle, circular graph nodes, orbital rings, decorative dots, enclosing badge, shadows, 3D or textures. Single centered symbol on a square canvas, moderate padding. Genuinely transparent alpha background, no white square. Professional and restrained, not a clip-art molecule or generic network icon.

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
