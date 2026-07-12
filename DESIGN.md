---
name: Apex Analytics
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#adc6ff'
  on-secondary: '#002e6a'
  secondary-container: '#0566d9'
  on-secondary-container: '#e6ecff'
  tertiary: '#d0bcff'
  on-tertiary: '#3c0091'
  tertiary-container: '#b090ff'
  on-tertiary-container: '#4600a7'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#d8e2ff'
  secondary-fixed-dim: '#adc6ff'
  on-secondary-fixed: '#001a42'
  on-secondary-fixed-variant: '#004395'
  tertiary-fixed: '#e9ddff'
  tertiary-fixed-dim: '#d0bcff'
  on-tertiary-fixed: '#23005c'
  on-tertiary-fixed-variant: '#5516be'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '800'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1.4'
    letterSpacing: 0.1em
  data-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: -0.01em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
  container-max: 1440px
---

## Brand & Style

The design system is engineered for elite performance, high-stakes decision-making, and immersive sports storytelling. It channels the energy of professional football through a high-contrast, dark-mode aesthetic that prioritizes clarity and visual impact.

The style is a fusion of **Corporate Modern** and **Glassmorphism**. It utilizes deep, ink-like canvases to provide a stable foundation for vibrant, luminescent data visualization. This design system evokes a sense of "Command Center" sophistication—precise, authoritative, and technologically advanced. Visual weight is managed through translucency and razor-sharp borders rather than heavy shadows, ensuring the UI feels lightweight and fast-reacting.

## Colors

The palette is anchored by **Deep Charcoal (#0f172a)** for surfaces and an even darker pitch for the global background. This creates a receding space where content can advance forward.

- **Primary Emerald (#10b981):** Used for positive trends, active states, and success metrics.
- **Electric Blue (#3b82f6):** Utilized for secondary data sets, links, and interactive elements.
- **System Accents:** A spectrum of vibrant violets and oranges is reserved strictly for complex multi-series charts to ensure distinct categorization.
- **High-Contrast Text:** White (#f8fafc) is used exclusively for critical headlines and primary data points, while Slate (#94a3b8) is used for labels and metadata to reduce visual noise.

## Typography

The design system utilizes **Inter** as the primary typeface for its exceptional legibility and neutral, geometric architecture. For data-heavy contexts, **JetBrains Mono** is introduced as a supporting label font to provide a technical, precision-oriented feel.

- **Headlines:** Use tight letter-spacing and bold weights to mimic sports broadcast graphics.
- **Data Display:** Large numerical figures should always use tabular lining to ensure numbers align vertically in columns.
- **Labels:** Small caps and increased letter spacing are used for tertiary metadata (e.g., "MATCH MINUTE" or "XG PROBABILITY") to distinguish them clearly from interactive body text.

## Layout & Spacing

This design system employs a **Fluid Grid** model built on a 4px baseline unit. The layout is designed to maximize information density without sacrificing clarity.

- **Dashboard Grid:** A 12-column layout with 24px gutters. Dashboard "widgets" or cards should span 3, 4, 6, or 12 columns depending on the complexity of the visualization.
- **Density:** High density. White space is used strategically around key data points, but the overall interface is packed with utility to minimize scrolling for professional analysts.
- **Responsive Behavior:** On mobile, columns collapse to a single stack. On ultra-wide displays, the content width is capped at 1440px to maintain comfortable scan lines for text-heavy tables.

## Elevation & Depth

Depth is established through **Tonal Layers** and **Subtle Outlines** rather than traditional drop shadows.

- **Base Layer:** The darkest shade, representing the global background.
- **Card Layer:** Deep charcoal surfaces with a 1px solid border (#1e293b). 
- **Glass Layer:** Used for floating headers or modal overlays. These use a `backdrop-filter: blur(12px)` and a semi-transparent background (`rgba(15, 23, 42, 0.7)`).
- **Interaction Depth:** When an element is hovered or active, its border color shifts to the Primary Emerald or Electric Blue to signify elevation, rather than moving the element's Y-axis position.

## Shapes

The shape language is **Rounded**, balancing the aggressive nature of sports data with modern UI approachability. 

- **Primary Containers:** 0.5rem (8px) corner radius. This applies to cards, input fields, and large buttons.
- **Inner Elements:** Nested elements (like chips inside cards) use a smaller radius (4px) to maintain visual harmony.
- **Data Markers:** Points on line charts or small status indicators remain sharp or perfectly circular to emphasize precision.

## Components

### Buttons & Inputs
- **Primary Action:** Solid Emerald background with black text for maximum "click" contrast.
- **Ghost Action:** Transparent background with a 1px Slate border; icon-only or text-only for secondary navigation.
- **Inputs:** Darker than the card surface, using an inset appearance with a focus state that highlights the entire border in Electric Blue.

### Data Visualization
- **Charts:** Use thin 1px lines for axes in muted Slate. Grid lines should be minimal or omitted.
- **Tooltips:** Always glassmorphic. When hovering over a data point, the tooltip should appear with a high-blur background to sit "above" the chart.

### Cards & Lists
- **Stat Cards:** Feature a large "Data-LG" headline for the primary metric, with a small Sparkline visualization at the bottom or a percentage change indicator in the top right.
- **Data Tables:** Zebra-striping is avoided; instead, use 1px horizontal dividers. The header row should be styled with "Label-Caps" typography and a slightly lighter surface background.

### Chips & Badges
- **Status Badges:** Small, pill-shaped markers. Use a low-opacity version of the accent color for the background (e.g., 10% Emerald) and the full-saturation color for the text.