# Brasskey brand kit

The Brasskey logo is the lowercase word **brasskey** in Instrument Sans Bold, letter spacing
-1%, followed by a solid cursor block. Letters are converted to vector outlines, so no font is
needed to display the logo.

## Files

| File | Use |
| --- | --- |
| `logo/brasskey-wordmark-on-dark.svg` | Main logo on dark backgrounds (white letters, Patina cursor) |
| `logo/brasskey-wordmark-on-light.svg` | Main logo on light backgrounds (black letters, Deep patina cursor) |
| `logo/brasskey-wordmark-mono-black.svg` | One-color black (print, stamps, single-color uses) |
| `logo/brasskey-wordmark-mono-white.svg` | One-color white (dark photos, single-color uses) |
| `icon/brasskey-icon-square.svg` | GitHub and social avatars (platforms crop to a circle themselves) |
| `icon/brasskey-icon-rounded.svg` | App-style icon with rounded corners |
| `icon/favicon.svg`, `icon/favicon.ico` | Website favicon (the .ico holds 16, 32 and 48 px) |
| `icon/favicon-16.svg` | 16 px favicon: the letter b only (a cursor would blur at this size) |
| `png/` | PNG versions: wordmarks at 600 and 1200 px wide, icons at 64 to 512 px, `apple-touch-icon-180.png` |
| `social/brasskey-social-1280x640.png` | GitHub social preview and link-sharing image |

## Colors

| Name | Hex | Use |
| --- | --- | --- |
| Ink | #000000 | Letters on light, dark backgrounds |
| Paper | #FFFFFF | Letters on dark, light backgrounds |
| Patina | #2BB3A3 | Cursor and accents on dark backgrounds |
| Deep patina | #0E7C70 | Cursor and accents on light backgrounds (meets contrast on white) |
| Graphite | #5B6470 | Secondary text |

## Type

- Logo: custom outlines from Instrument Sans Bold (SIL Open Font License 1.1; license in
  `licenses/OFL-InstrumentSans.txt`).
- Website and docs text: Instrument Sans.
- Command-line examples: a monospace font such as JetBrains Mono.

## Usage rules

- Clear space: keep empty space around the logo at least the width of the cursor block.
- Minimum size: wordmark at least 96 px wide on screens; below that, use the icon.
- Use the on-dark file on dark backgrounds and the on-light file on light ones.
- Do not change the letter spacing, retype the word in a font, stretch, rotate, outline, add
  shadows or gradients, or recolor individual letters.
- Do not separate the cursor from the word, except in the icon, where "b" plus cursor stands in
  for the full name.
- Write the name as "Brasskey" in sentences and `brasskey` in code.

## README header that follows light and dark mode on GitHub

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/logo/brasskey-wordmark-on-dark.svg">
  <img alt="Brasskey" src="brand/logo/brasskey-wordmark-on-light.svg" width="320">
</picture>
```

## Trademark

BRASSKEY is pending trademark filing by Prcept AI LLP. Use the TM symbol until registration.
