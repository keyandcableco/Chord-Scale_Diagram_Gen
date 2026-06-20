# Chord & Scale Diagram Generator -- Web GUI

A local web form for chord_diagram.py. Spell a chord or scale, pick
your options, and download keyboard diagrams, guitar fretboard
diagrams, and a notated PDF -- no command line needed.

## Setup

Requires Python 3 and Flask:

    pip install flask --break-system-packages

(omit `--break-system-packages` if you're using a virtual environment)

For the final combined PDF, this also needs `lilypond-book` and
`pdflatex` on your system PATH, same as the command-line version. If
those aren't installed, you'll still get a .zip of every keyboard
diagram, fretboard diagram, chart, and LilyPond source file generated
-- just not the final merged PDF.

## Running

    python3 app.py

Then open http://127.0.0.1:5000 in your browser.

Press Ctrl+C in the terminal to stop the server when you're done.

## What you can generate

### Chords

Type a root-position spelling (e.g. `Ab C Eb`) and choose:

- **Inversions** -- root position through however many inversions the
  chord has, each on its own keyboard diagram and its own line of
  notation.
- **Full chord view** -- every occurrence of every chord tone repeated
  across 5 octaves on the keyboard.
- **Diatonic key & function info** -- every practical major/minor key
  this chord belongs to, and its Roman-numeral role in each (e.g. G
  minor shown as ii in F major, iii in Eb major, vi in Bb major).
- **Guitar fretboard (CAGED shapes)** -- all 5 movable shapes (C, A,
  G, E, D forms), correctly transposed to the chord's root. Major and
  minor triads only -- this is a fixed-shape system, so it doesn't
  apply to diminished, augmented, or seventh chords.
- **Full fretboard** -- every occurrence of every chord tone across
  all 6 strings and 12 frets, labeled by letter name. Works for any
  chord, including sevenths and other qualities CAGED can't represent.

### Scales

Type a tonic-first spelling (e.g. `C D E F G A B C`) and choose:

- **Key signature staff** -- a small staff showing just the clef and
  key signature, using whichever of LilyPond's built-in modes matches
  your scale. For scales that don't match a mode exactly (like
  harmonic minor), it shows the closest natural key signature and
  notes the extra accidental separately, rather than guessing at a key
  signature that doesn't really exist.
- **Full guitar fretboard** -- every scale tone across all 6 strings
  and 12 frets, labeled by letter name.
- **Harmonized chord chart** -- builds a triad (or seventh chord) on
  every scale degree and charts the actual chromatic intervals between
  them, so you can see at a glance which chords are major, minor,
  diminished, etc.
- **Render every harmonized chord** -- once you've built the chart,
  each of its chords can be expanded into its own full set of
  inversions, full chord view, diatonic info, CAGED shapes, and full
  fretboard -- exactly as if you'd typed that chord in by hand.

## Output

If `lilypond-book` and `pdflatex` are both available, you get one
finished PDF with everything laid out, each chord or scale on its own
page. If not, you get a .zip with every individual keyboard diagram,
fretboard diagram, chart, and the LilyPond source files, so you can
finish the PDF yourself once those tools are installed.

## Notes

- Runs entirely on your machine; nothing is sent anywhere.
- Each click of "Generate PDF" runs independently -- the form doesn't
  remember previous chords, matching the simpler one-document-per-run
  workflow.
- Want multiple chords or scales combined into one document -- say, a
  whole harmonized scale's chords alongside a few extra chords you
  spell by hand, all in one PDF? The command-line version
  (`chord_diagram.py`, run directly with `python3 chord_diagram.py`)
  supports this: after each chord or scale, it asks "Add another
  chord or scale?" and keeps looping until you say no, then builds
  everything into a single combined document. The web form here is
  intentionally simpler -- one chord or scale, one document, one
  click -- so if you want that extra flexibility, use the CLI
  directly instead.
- This is meant for one person using it locally. It is not hardened
  for being exposed on a network or used by multiple people at once.


## To-Do

- Sharp-keys default filename doesn't distinguish *sharp* at all.
- Center scale staff when key signature engraver is present.
- Perhaps highlight tonic note on fretboard layout.
- Remove b11 and the upper 5 and 7 from the harmonized chords chart
- CAGED shapes not showing extended sevenths? Root probably dominates
- Diatonic functions incorrect for Maj7 as a V
