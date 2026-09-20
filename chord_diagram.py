#!/usr/bin/env python3
"""
chord_diagram.py

Interactive tool for generating chord/scale teaching materials.
Now produces one page per chord/scale with its own title.
"""

import os
import re
import shutil
import subprocess
import sys
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle


# ----------------------------------------------------------------------
# Note-name parsing
# ----------------------------------------------------------------------

LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
PITCH_CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
MAJOR_SCALE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

LY_SHARP = "is"
LY_FLAT = "es"

# Maximum width (mm) the main notation staff is allowed to use before
# wrapping to a new line -- with ragged-right = ##t (see
# build_single_lilypond) this is a CEILING, not a forced stretch
# target: short chords/scales render at their natural width and don't
# get stretched out to fill this whole span. Set comfortably inside
# this document's real LaTeX text width (US Letter, 1in margins ->
# ~165mm) so nothing here can overflow the page.
PAPER_LINE_WIDTH_MM = 160

# How many octaves the "full scale across the keyboard" view repeats
# the scale's tones over. Chords default to 5 octaves for their full
# view, but a 7-note scale repeated 5x is a lot of crowded dots --
# 3 is enough to show the repeating pattern clearly without the
# diagram becoming unreadable.
FULL_SCALE_KEYBOARD_OCTAVES = 3


def parse_note(token):
    token = token.strip()
    if not token:
        raise ValueError("Empty note name.")

    letter = token[0].upper()
    if letter not in PITCH_CLASS:
        raise ValueError(f"'{token}': notes must start with A-G.")

    rest = token[1:].strip().lower()
    rest = rest.replace("flat", "b").replace("sharp", "#")

    accidental = 0
    display_accidental = ""
    for ch in rest:
        if ch == 'b':
            accidental -= 1
            display_accidental += "b"
        elif ch == '#':
            accidental += 1
            display_accidental += "#"
        else:
            raise ValueError(f"'{token}': unrecognized accidental '{ch}'.")

    pc = (PITCH_CLASS[letter] + accidental) % 12
    display_name = letter + display_accidental
    return letter, accidental, pc, display_name


def to_lilypond_pitch(letter, accidental):
    name = letter.lower()
    if accidental > 0:
        name += LY_SHARP * accidental
    elif accidental < 0:
        name += LY_FLAT * (-accidental)
    return name


# ----------------------------------------------------------------------
# Octave assignment, degree labeling, inversions, etc.
# (unchanged from original)
# ----------------------------------------------------------------------

def assign_octaves(parsed_notes, start_octave=4, mode='chord'):
    """
    Assign a concrete octave to each note in a scale or chord spelling,
    starting from start_octave for the first note.

    mode='chord' (default): always ascend -- each note is placed in
    the lowest octave strictly above the previous note's absolute
    pitch. This is correct for chord voicings, where the user's
    spelling order always implies an ascending stack (F Ab Eb Bb means
    Eb above Ab, Bb above Eb, not the other way around). The original
    pre-nearest-neighbor behavior.

    mode='scale': use nearest-neighbor -- each note is placed in
    whichever octave puts it closest to the previous note, including
    below it if that's nearer. This is correct for scales that include
    intentional descending chromatic steps (e.g. the Flamenco scale
    E F G# G A B C D D#, where G is one semitone below G# and should
    stay in the same octave rather than being bumped an octave up).
    For monotonically ascending scales, nearest-neighbor gives the
    same result as always-ascending.
    """
    result = []
    prev_absolute = None
    for letter, accidental, pc, display_name in parsed_notes:
        if prev_absolute is None:
            absolute = start_octave * 12 + pc
        elif mode == 'scale':
            # Nearest-neighbor: pick the octave that puts this pc
            # closest to the previous note, above or below.
            base_octave = prev_absolute // 12
            same = base_octave * 12 + pc
            above = same + 12
            below = same - 12
            dist_same = abs(same - prev_absolute)
            dist_above = abs(above - prev_absolute)
            dist_below = abs(below - prev_absolute)
            best_dist = min(dist_same, dist_above, dist_below)
            if dist_above == best_dist:
                absolute = above
            elif dist_same == best_dist:
                absolute = same
            else:
                absolute = below
        else:
            # Always-ascending: place in the lowest octave strictly
            # above the previous note.
            candidate_octave = prev_absolute // 12
            absolute = candidate_octave * 12 + pc
            while absolute <= prev_absolute:
                candidate_octave += 1
                absolute = candidate_octave * 12 + pc
        result.append({
            "display_name": display_name,
            "letter": letter,
            "accidental": accidental,
            "pc": pc,
            "octave": absolute // 12,
        })
        prev_absolute = absolute
    return result


def degree_label(root_letter, root_accidental, note_letter, note_accidental,
                  above_octave=False):
    """
    Return the interval label for a note relative to a root, e.g. 'b7',
    '#5', 'b2'. When above_octave=True, simple degrees that have
    conventional extension names are promoted: 2->9, b2->b9, #2->#9,
    4->11, #4->#11, 6->13, b6->b13. This reflects that the note is
    voiced above the octave relative to the chord root -- a b9 and a
    b2 are the same pitch class but different voicings; the keyboard
    diagram should show which one the user actually meant.
    """
    EXTENSION_MAP = {
        'b2': 'b9', '2': '9', '#2': '#9',
        '4': '11', '#4': '#11',
        'b6': 'b13', '6': '13',
    }
    li_root = LETTERS.index(root_letter)
    li_note = LETTERS.index(note_letter)
    base_degree = ((li_note - li_root) % 7) + 1

    root_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
    note_pc = (PITCH_CLASS[note_letter] + note_accidental) % 12
    semitone_distance = (note_pc - root_pc) % 12
    expected_semitones = MAJOR_SCALE_SEMITONES[base_degree]
    diff = semitone_distance - expected_semitones
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12

    if diff == 0:
        prefix = ""
    elif diff == -1:
        prefix = "b"
    elif diff == -2:
        prefix = "bb"
    elif diff == 1:
        prefix = "#"
    elif diff == 2:
        prefix = "##"
    else:
        prefix = f"({diff:+d})"

    label = f"{prefix}{base_degree}"
    if above_octave:
        label = EXTENSION_MAP.get(label, label)
    return label


def degree_labels_for_voicing(notes_with_octaves, root_letter, root_accidental,
                               extension_pcs=None):
    """
    Return interval labels for each note in a voicing (e.g. ['1', '3',
    'b7', 'b9']). Notes whose pitch class is in extension_pcs are
    labeled as extensions (b9 not b2, 9 not 2, 11 not 4, 13 not 6).

    extension_pcs: set of pitch classes (0-11) that should be labeled
    as extensions. If None, infers from the voicing itself -- notes
    whose absolute pitch is more than one octave above the root at
    octave 4. Pass this explicitly when labeling inversions, since the
    root-position spelling (not the inversion's bass note) determines
    the harmonic register of each tone.
    """
    root_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
    root_ref_abs = 4 * 12 + root_pc

    if extension_pcs is None:
        # Infer from this voicing's own absolute pitches -- correct for
        # root-position voicings, but not for inversions.
        extension_pcs = {
            n["pc"] for n in notes_with_octaves
            if (n["octave"] * 12 + n["pc"]) - root_ref_abs > 12
        }

    labels = []
    for n in notes_with_octaves:
        above_octave = n["pc"] in extension_pcs
        labels.append(
            degree_label(root_letter, root_accidental,
                         n["letter"], n["accidental"],
                         above_octave=above_octave)
        )
    return labels


def rotate(lst, k):
    return lst[k:] + lst[:k]


def generate_inversions(parsed_root_notes, how_many):
    inversions = []
    for k in range(how_many):
        rotated = rotate(parsed_root_notes, k)
        inversions.append(assign_octaves(rotated, start_octave=4))
    return inversions


INVERSION_NAMES = [
    "Root pos.", "First inv.", "Second inv.",
    "Third inv.", "Fourth inv.", "Fifth inv.",
    "Sixth inv.", "Seventh inv.",
]


def inversion_name(k):
    if k < len(INVERSION_NAMES):
        return INVERSION_NAMES[k]
    return f"Inversion {k}"


def generate_full_chord_voicing(parsed_root_notes, n_octaves, start_octave=4):
    """
    Repeat the chord tones across n_octaves "rungs", root lowest in each
    rung, ascending from there -- e.g. for Ab/C/Eb: Ab3 C4 Eb4, Ab4 C5 Eb5,
    Ab5 C6 Eb6, ...

    IMPORTANT: this must NOT just stamp the same octave number onto every
    note in root-position order. Doing that put C and Eb (lower pitch
    classes than Ab) BELOW the root within the labeled "octave 4" group,
    since Ab's pitch class (8) is higher than C's (0) -- so the root
    ended up as the highest note of its own group rather than the
    lowest, both on the keyboard image and in the LilyPond chord. Reusing
    assign_octaves' proven ascending-from-the-first-note logic for each
    rung, then continuing the next rung from where the previous one left
    off, keeps the root anchored at the bottom of every group.
    """
    result = []
    prev_absolute = None
    for _ in range(n_octaves):
        if prev_absolute is None:
            rung = assign_octaves(parsed_root_notes, start_octave=start_octave)
        else:
            # Anchor this rung's root just above the previous rung's
            # last (highest) note, then ascend from there as usual.
            next_octave_guess = prev_absolute // 12
            rung = assign_octaves(parsed_root_notes, start_octave=next_octave_guess)
            # If that landed at or below the previous note (e.g. root's
            # pitch class is low), bump up an octave and retry.
            while rung[0]["octave"] * 12 + rung[0]["pc"] <= prev_absolute:
                next_octave_guess += 1
                rung = assign_octaves(parsed_root_notes, start_octave=next_octave_guess)
        result.extend(rung)
        last = rung[-1]
        prev_absolute = last["octave"] * 12 + last["pc"]
    return result


# ----------------------------------------------------------------------
# LilyPond - ONE SCORE PER CHORD/SCALE
# ----------------------------------------------------------------------

def lilypond_octave_marks(octave):
    diff = octave - 3
    return "'" * diff if diff >= 0 else "," * (-diff)


def ly_chord_block(label_text, notes_with_octaves, above=True):
    pitches = []
    for n in notes_with_octaves:
        ly_pitch = to_lilypond_pitch(n["letter"], n["accidental"])
        ly_octave = lilypond_octave_marks(n["octave"])
        pitches.append(f"{ly_pitch}{ly_octave}")
    chord_token = "<" + " ".join(pitches) + ">1"
    safe_label = label_text.replace('"', r'\"')
    direction = "^" if above else "_"
    return f'{chord_token}{direction}\\markup {{ \\bold "{safe_label}" }}\n    \\bar "||"'


def ly_scale_block(label_text, notes_with_octaves, above=True):
    pitches = []
    for n in notes_with_octaves:
        ly_pitch = to_lilypond_pitch(n["letter"], n["accidental"])
        ly_octave = lilypond_octave_marks(n["octave"])
        pitches.append(f"{ly_pitch}{ly_octave}4")
    safe_label = label_text.replace('"', r'\"')
    direction = "^" if above else "_"
    first_note = pitches[0] if pitches else "c'"
    rest = " ".join(pitches[1:])
    return f'{first_note}{direction}\\markup {{ \\bold "{safe_label}" }} {rest}\n    \\bar "||"'


# ----------------------------------------------------------------------
# Key signature staff: detect which of LilyPond's built-in church modes
# (if any) matches the scale, and build a key-signature-only staff
# block using \key <tonic> \<mode>. LilyPond's own key-signature
# engraving already gets sharps/flats in the correct conventional
# order for free -- this just has to identify the right tonic + mode
# to hand it, never compute sharp/flat counts or ordering by hand.
# ----------------------------------------------------------------------

CHURCH_MODES = {
    'ionian':     [0, 2, 4, 5, 7, 9, 11],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'phrygian':   [0, 1, 3, 5, 7, 8, 10],
    'lydian':     [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian':    [0, 2, 3, 5, 7, 8, 10],
    'locrian':    [0, 1, 3, 5, 6, 8, 10],
}

# LilyPond also accepts plain \major / \minor as synonyms for
# ionian / aeolian -- using those names specifically (rather than
# \ionian / \aeolian) reads more familiarly for the common case.
MODE_DISPLAY_NAME = {
    'ionian': 'major', 'aeolian': 'minor', 'dorian': 'dorian',
    'phrygian': 'phrygian', 'lydian': 'lydian',
    'mixolydian': 'mixolydian', 'locrian': 'locrian',
}
MODE_LY_KEYWORD = {
    'ionian': 'major', 'aeolian': 'minor', 'dorian': 'dorian',
    'phrygian': 'phrygian', 'lydian': 'lydian',
    'mixolydian': 'mixolydian', 'locrian': 'locrian',
}


def detect_mode_and_key(scale_notes):
    """
    Compare the scale's interval pattern (from its own tonic) against
    each of the seven church modes. Returns:
      (tonic_letter, tonic_accidental, mode_name, differences)
    where differences is a list of (degree_1indexed, semitone_diff)
    for any degree that doesn't match the closest mode -- empty if the
    scale matches a mode exactly. semitone_diff is actual minus
    expected, e.g. +1 means that degree is a semitone higher than the
    matched mode calls for (a "raised" degree).

    Only scales of exactly 7 notes are matched against a mode (the
    church modes are inherently a 7-degree concept); for any other
    length, mode_name comes back as None and the caller should skip
    drawing a key-signature staff.
    """
    if len(scale_notes) != 7:
        return None, None, None, None

    tonic_letter, tonic_accidental = scale_notes[0][0], scale_notes[0][1]
    tonic_pc = (PITCH_CLASS[tonic_letter] + tonic_accidental) % 12
    intervals = [(PITCH_CLASS[n[0]] + n[1] - tonic_pc) % 12 for n in scale_notes]

    best_mode = None
    best_diffs = None
    for mode_name, steps in CHURCH_MODES.items():
        diffs = [(i + 1, intervals[i] - steps[i]) for i in range(7) if intervals[i] != steps[i]]
        if best_diffs is None or len(diffs) < len(best_diffs):
            best_mode, best_diffs = mode_name, diffs

    return tonic_letter, tonic_accidental, best_mode, best_diffs


def describe_degree_difference(degree, semitone_diff):
    """Plain-language description of one scale degree's deviation from
    the matched mode, e.g. 'raised 7th degree'."""
    # normalize wraparound (e.g. a -11 should read as +1)
    diff = semitone_diff
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12

    ordinal = {1: '1st', 2: '2nd', 3: '3rd'}.get(degree, f"{degree}th")
    if diff == 1:
        return f"raised {ordinal} degree"
    elif diff == -1:
        return f"lowered {ordinal} degree"
    elif diff == 2:
        return f"raised {ordinal} degree (by two semitones)"
    elif diff == -2:
        return f"lowered {ordinal} degree (by two semitones)"
    else:
        return f"altered {ordinal} degree"


def key_signature_block(scale_notes):
    """
    Build a standalone LilyPond \\score {...} block: just a clef and
    key signature, no notes, no time signature, on its own staff.
    Returns (score_block_text, info_message) where info_message
    describes any accidental(s) needed beyond what the printed key
    signature already covers (e.g. harmonic minor's raised 7th) -- or
    None if the scale matches a mode exactly.

    Returns (None, None) if the scale isn't exactly 7 notes, since the
    church modes (and therefore a single conventional key signature)
    are inherently a 7-degree idea -- a 5- or 6-note scale doesn't have
    one conventional key signature to fall back on.

    score_block_text is just the \\score {...} text -- no \\version or
    \\header -- so it composes cleanly as one of build_single_lilypond's
    extra_score_blocks, appended after the main scale/chord \\score in
    the same .ly file.
    """
    tonic_letter, tonic_accidental, mode_name, diffs = detect_mode_and_key(scale_notes)
    if mode_name is None:
        return None, None

    ly_tonic = to_lilypond_pitch(tonic_letter, tonic_accidental)
    ly_mode = MODE_LY_KEYWORD[mode_name]
    mode_display = MODE_DISPLAY_NAME[mode_name]
    tonic_display = key_display_name(tonic_letter, tonic_accidental)

    score_block = f'''\\markup {{ \\bold "Key Signature" }}
        
        \\score {{
          \\new Staff \\with {{
            \\remove "Time_signature_engraver"
          }} {{
            \\clef treble
            \\key {ly_tonic} \\{ly_mode}
            \\cadenzaOn
            s1 s1 s1 s1 s1 s1
          }}
          \\layout {{
            ragged-right = ##t
            indent = 0\\mm
            line-width = 70\\mm
          }}
        }}'''

    if diffs:
        descriptions = [describe_degree_difference(d, s) for d, s in diffs]
        info_message = (
            f"Key signature shown is {tonic_display} {mode_display} "
            f"(closest match). This scale also has a {', and a '.join(descriptions)} "
            f"not shown in the key signature -- written as an accidental in the music."
        )
    else:
        info_message = None

    return score_block, info_message


def build_single_lilypond(title, blocks, extra_score_blocks=None):
    """
    Build LilyPond source for one chord or scale.

    extra_score_blocks: optional list of complete, independent
    \\score {...} block strings to append after the main one -- used
    for the key-signature-only staff, which needs its own \\new Staff
    context with a different engraver setup (no time signature
    engraver) and so can't just be another music line inside the main
    staff alongside the scale's own notes.

    IMPORTANT: when there's more than one \\score block, they must be
    wrapped in an explicit \\book {...} block. Inside a lilypond-book
    \\begin{lilypond}...\\end{lilypond} snippet specifically (as opposed
    to a plain standalone .ly file compiled directly with `lilypond`),
    LilyPond's own documentation states that only the FIRST \\score or
    \\markup in the file is rendered unless everything is wrapped in an
    explicit \\book -- without this wrapper the key signature staff
    would compile without error but silently never appear in the
    final PDF.

    Each \\score sets its own line-width (the main staff wants to be
    wider than the little key-signature-only staff). Several earlier
    attempts at this fix tried to get LilyPond's own margin/centering
    math to land the system in the middle of the declared line-width
    (which was itself meant to match the real LaTeX page) -- but a
    real rendered PDF still showed the main staff sitting well right
    of center (measured ~25mm off on a US Letter page), while the
    keyboard/fretboard images (centered via plain LaTeX \\begin{center}
    on a normal image, no LilyPond involved) landed dead center.

    That comparison is the key clue: LaTeX's own \\begin{center} DOES
    work correctly here -- it's just centering a LilyPond-rendered
    image whose own bounding box has asymmetric whitespace baked into
    it. With ragged-right = ##f (the old setting), LilyPond stretches
    the system to fill the FULL declared line-width regardless of how
    much actual music is in it -- so the rendered image's bounding box
    is that whole stretched line-width, and if the notes don't end up
    sitting symmetrically inside it (e.g. because of indent, or just
    how the engraver placed things), centering that box centers the
    box, not the notes inside it.

    The fix: ragged-right = ##t makes each system crop to its natural
    width instead of stretching to fill line-width -- the same "tight
    bounding box" approach the keyboard/fretboard PNGs already use
    (matplotlib's bbox_inches='tight'), which is exactly why THOSE
    already centered correctly. line-width is kept as a maximum/wrap
    width, not a forced stretch target, so a short chord/scale doesn't
    accidentally get stretched out across the whole page width.
    """
    safe_title = title.replace('"', r'\"')
    body = "\n    ".join(blocks)
    main_score = f'''\\score {{
  \\new Staff {{
    \\clef treble
    \\time 4/4
    {body}
  }}
  \\layout {{
    ragged-right = ##t
    indent = 0\\mm
    line-width = {PAPER_LINE_WIDTH_MM}\\mm
  }}
}}'''

    paper_block = f'''\\paper {{
  indent = 0\\mm
  line-width = {PAPER_LINE_WIDTH_MM}\\mm
}}'''

    if extra_score_blocks:
        extras = "\n\n".join(extra_score_blocks)
        all_scores = main_score + "\n\n" + extras
        return f'''\\version "2.24.0"

#(set-global-staff-size 26)

\\header {{
  title = "{safe_title}"
  tagline = ##f
}}

\\book {{
{paper_block}

{all_scores}
}}
'''

    return f'''\\version "2.24.0"

#(set-global-staff-size 26)

\\header {{
  title = "{safe_title}"
  tagline = ##f
}}

{paper_block}

{main_score}
'''


# ----------------------------------------------------------------------
# Keyboard diagram (unchanged)
# ----------------------------------------------------------------------

WHITE_FILL = "#FFFFFF"
WHITE_STROKE = "#999999"
BLACK_FILL = "#2C2C2A"
BLACK_STROKE = "#1A1A19"
HL_FILL = "#D85A30"
HL_STROKE = "#993C1D"
HL_TEXT = "#FFFFFF"
LABEL_TEXT = "#444441"

KEY_W = 1.0
WHITE_H = 4.0
BLACK_W = 0.62
BLACK_H = 2.5
WHITE_LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
WHITE_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
BLACK_KEYS = [(1.0, 1), (2.0, 3), (4.0, 6), (5.0, 8), (6.0, 10)]

# White keys are numbered absolutely: index = octave * 7 + position in
# WHITE_LETTERS, so C4 -> 28, F4 -> 31. These two tables give the white
# key at or below / at or above any pitch class, which is how a chord's
# pitch range is turned into a range of keys to draw.
WHITE_BELOW_PC = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}
WHITE_ABOVE_PC = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}
# A white key has a black key immediately above it except after E and B.
WHITE_HAS_BLACK_ABOVE = {'C', 'D', 'F', 'G', 'A'}
# Keyboards start and end on one of these two letters. Both begin a
# black-key group (C starts the pair, F starts the triplet), so a window
# anchored to either still reads as a piece of a real keyboard rather
# than an arbitrary slice.
KEYBOARD_ANCHOR_POSITIONS = (0, 3)   # C and F within WHITE_LETTERS


def keyboard_white_span(notes_with_octaves, min_octaves=2):
    """
    Work out which white keys a voicing needs, returning
    (start_white_index, n_white_keys).

    The diagram used to always start on C and cover whole octaves from
    the lowest note's octave, so a chord like Gb7#11 -- which spans
    Gb up to a C two octaves later -- forced three full octaves of
    keyboard, most of it empty, and shrank the key labels to nothing.
    Instead the start is snapped down to the nearest anchor (C or F)
    below the lowest note and the width is a whole number of octaves
    from there, so Gb7#11 runs F to E across fourteen keys instead of
    twenty-one.

    Whole-octave widths mean a window ends on the letter below its
    start: C to B, or F to E. The width never falls below min_octaves.
    """
    lowest = min(n["octave"] * 12 + n["pc"] for n in notes_with_octaves)
    highest = max(n["octave"] * 12 + n["pc"] for n in notes_with_octaves)

    start = (lowest // 12) * 7 + WHITE_BELOW_PC[lowest % 12]
    end = (highest // 12) * 7 + WHITE_ABOVE_PC[highest % 12]

    while start % 7 not in KEYBOARD_ANCHOR_POSITIONS:
        start -= 1

    n_white = max(end - start + 1, 7 * max(1, min_octaves))
    # Round up to whole octaves from the start key.
    if n_white % 7:
        n_white += 7 - (n_white % 7)
    return start, n_white


def draw_keyboard(notes_with_octaves, degree_labels, base_filename, min_octaves=2,
                  n_white_keys=None):
    """
    n_white_keys: force a minimum width in white keys. Inversions of one
    chord are drawn at a shared width so they can be compared, since
    each inversion on its own would otherwise pick a different window.
    """
    start_white, n_white = keyboard_white_span(notes_with_octaves, min_octaves)
    if n_white_keys and n_white_keys > n_white:
        n_white = n_white_keys

    # Keys are identified by absolute (pitch class, octave) now that the
    # window no longer starts at C in the lowest octave.
    highlighted = {}
    for n, lbl in zip(notes_with_octaves, degree_labels):
        highlighted[(n["pc"], n["octave"])] = (n["display_name"], lbl)

    label_positions = {}

    white_keys = []
    for i in range(n_white):
        w = start_white + i
        letter = WHITE_LETTERS[w % 7]
        white_keys.append((letter, w // 7))
    total_w = n_white * KEY_W

    fig_height_in = 3.3
    fig_w = max(fig_height_in * (total_w / WHITE_H) * 0.78, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_height_in), dpi=200)

    # White keys
    for i, (name, octave) in enumerate(white_keys):
        x = i * KEY_W
        pc = WHITE_PC[name]
        key = (pc, octave)
        is_hl = key in highlighted
        fill = HL_FILL if is_hl else WHITE_FILL
        stroke = HL_STROKE if is_hl else WHITE_STROKE
        lw = 2.2 if is_hl else 1.2
        rect = FancyBboxPatch((x, 0), KEY_W, WHITE_H,
                              boxstyle="round,pad=0,rounding_size=0.06",
                              linewidth=lw, edgecolor=stroke, facecolor=fill, zorder=1)
        ax.add_patch(rect)
        text_color = HL_TEXT if is_hl else LABEL_TEXT
        weight = "bold" if is_hl else "normal"
        label_in_key = highlighted[key][0] if is_hl else name
        ax.text(x + KEY_W / 2, 0.4, label_in_key, ha='center', va='center',
                fontsize=19, color=text_color, fontweight=weight, zorder=3)
        if is_hl:
            label_positions[key] = x + KEY_W / 2

    # Black keys sit on the boundary between two white keys, so they are
    # placed by walking the drawn white keys rather than by octave -- the
    # window can start on any anchor and end mid-pattern.
    for i, (name, octave) in enumerate(white_keys):
        if name not in WHITE_HAS_BLACK_ABOVE:
            continue
        if i == n_white - 1:
            continue        # no room past the last white key
        x = (i + 1) * KEY_W - BLACK_W / 2
        pc = (WHITE_PC[name] + 1) % 12
        key = (pc, octave)
        is_hl = key in highlighted
        fill = HL_FILL if is_hl else BLACK_FILL
        stroke = HL_STROKE if is_hl else BLACK_STROKE
        lw = 2.2 if is_hl else 1.2
        rect = FancyBboxPatch((x, WHITE_H - BLACK_H), BLACK_W, BLACK_H,
                              boxstyle="round,pad=0,rounding_size=0.04",
                              linewidth=lw, edgecolor=stroke, facecolor=fill, zorder=2)
        ax.add_patch(rect)
        if is_hl:
            cx = x + BLACK_W / 2
            cy = WHITE_H - BLACK_H / 2
            ax.text(cx, cy, highlighted[key][0], ha='center', va='center',
                    fontsize=15, color=HL_TEXT, fontweight='bold', zorder=3)
            label_positions[key] = cx

    ax.set_xlim(-0.15, total_w + 0.15)
    ax.set_ylim(-0.15, WHITE_H + 0.75)
    ax.set_aspect('equal')
    ax.axis('off')

    label_y = WHITE_H + 0.42
    seen = set()
    for n, lbl in zip(notes_with_octaves, degree_labels):
        key = (n["pc"], n["octave"])
        if key in label_positions and key not in seen:
            ax.text(label_positions[key], label_y, lbl,
                    ha='center', va='center', fontsize=15,
                    color=LABEL_TEXT, fontweight='bold', zorder=4)
            seen.add(key)

    fig.tight_layout(pad=0.15)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)



# ----------------------------------------------------------------------
# Harmonized scale chart (chromatic box diagram of diatonic chords)
# ----------------------------------------------------------------------
#
# Each column represents one SEMITONE, not one scale-degree position --
# this is what makes interval size visually apparent. A major third
# (4 semitones) spans visibly more empty boxes than a minor third (3
# semitones), even when both chords have the same "1-3-5" scale-degree
# shape. Chord quality (major/minor/dim/aug, and for sevenths: maj7,
# dom7, min7, m7b5, dim7, etc.) is derived mechanically by measuring
# real semitone distances -- nothing here hardcodes "major scale"
# rules, so feeding in a minor scale, harmonic minor, or any other
# scale produces the musically correct chord qualities for THAT scale.

CHART_LABEL_TEXT = "#444441"
CHART_BOX_STROKE = "#999999"
CHART_EMPTY_FILL = "#FFFFFF"
CHART_HL_FILL = "#D85A30"
CHART_HL_STROKE = "#993C1D"
CHART_HL_TEXT = "#FFFFFF"

# Fixed chromatic interval names, independent of whatever specific scale
# is loaded -- this is a fact about distance-from-the-tonic in semitones,
# not about the scale itself. A pentatonic scale's tones simply occupy a
# subset of these 24 positions (e.g. degrees 1,2,3,5,6) and leave the
# rest dark, exactly like a 7-note scale leaves its 5 chromatic
# passing-tone positions dark. First octave uses plain diatonic degree
# names with flats for the chromatic in-between positions; second
# octave switches to conventional extension names (9/11/13 and their
# flat/sharp alterations) per standard chord-extension naming -- the 3rd
# and 5th an octave up keep their plain names (a 10th/12th isn't
# normally renamed), but the chromatic neighbors of 9/11/13 always use
# the alteration name (b9/#9/b11/#11/b13/#13), even where one of those
# (b11, #13) lands on the same pitch as an octave-up 3rd/7th would --
# in the second octave that position is always named as the extension,
# never as the plain degree, since this row is specifically about
# extension-tone vocabulary.
CHROMATIC_DEGREE_NAMES = {
    0: '1', 1: 'b2', 2: '2', 3: 'b3', 4: '3', 5: '4',
    6: 'b5', 7: '5', 8: 'b6', 9: '6', 10: 'b7', 11: '7',
    12: '1', 13: 'b9', 14: '9', 15: '#9', 16: 'b11', 17: '11',
    18: '#11', 19: '5', 20: 'b13', 21: '13', 22: '#13', 23: '7',
}

ROMAN_BASE = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
              'XI', 'XII']


def classify_triad(root_pc, third_pc, fifth_pc):
    """
    Classify a triad purely by measuring semitone distances from the
    root to the third and fifth -- no scale-type assumptions.

    The four standard triads (maj, min, dim, aug) are matched first.
    Beyond those, a broad table covers suspended chords, altered fifths,
    and other common three-note sonorities that appear when harmonizing
    non-diatonic or chromatic scales (where stacking every other scale
    tone can produce almost any interval combination). The table is
    organised by third type first (so the result is musically intuitive:
    "what kind of third, then what kind of fifth?"), then falls back to
    a plain interval description (e.g. "2+5") so the chart always shows
    something meaningful rather than "?" for unusual stacks.
    """
    third_int = (third_pc - root_pc) % 12
    fifth_int = (fifth_pc - root_pc) % 12
    table = {
        # Standard four
        (4, 7):  ("Major triad",      "maj"),
        (3, 7):  ("Minor triad",      "min"),
        (3, 6):  ("Diminished triad", "dim"),
        (4, 8):  ("Augmented triad",  "aug"),
        # Suspended chords (no 3rd -- 2nd or 4th instead)
        (2, 7):  ("Suspended 2nd",    "sus2"),
        (5, 7):  ("Suspended 4th",    "sus4"),
        (2, 5):  ("Sus2 flat 5",      "sus2b5"),
        (5, 6):  ("Sus4 flat 5",      "sus4b5"),
        (2, 8):  ("Sus2 aug 5",       "sus2#5"),
        (5, 8):  ("Sus4 aug 5",       "sus4#5"),
        # Minor/major third + perfect fourth (open quartal-ish voicings)
        (3, 5):  ("Minor add 4",      "minadd4"),
        (4, 5):  ("Major add 4",      "majadd4"),
        # Altered fifths on standard thirds
        (4, 6):  ("Major flat 5",     "majb5"),
        (3, 8):  ("Minor aug 5",      "min#5"),
        # Quartal / wide-interval stacks
        (5, 10): ("Quartal",          "qrt"),
        (5, 9):  ("Sus4 maj6",        "sus4maj6"),
        # Chromatic / semitone clusters -- name by the actual intervals
        # so the chart shows something legible rather than just "?"
        (1, 4):  ("Semitone + M3",    "1+4"),
        (1, 5):  ("Semitone + P4",    "1+5"),
        (1, 6):  ("Semitone + tt",    "1+6"),
        (1, 7):  ("Semitone + P5",    "1+7"),
        (1, 8):  ("Semitone + m6",    "1+8"),
        (2, 4):  ("M2 + M3",          "2+4"),
        (2, 6):  ("M2 + tt",          "2+6"),
        (4, 5):  ("M3 + P4",          "3+5"),
        (6, 10): ("Tritone + m7",     "tt+10"),
        (6, 9):  ("Tritone + M6",     "tt+9"),
        (6, 7):  ("Tritone + P5",     "tt+7"),
    }
    if (third_int, fifth_int) in table:
        return table[(third_int, fifth_int)]
    # Genuine last resort -- at least show the raw semitone distances
    # so the chart is informative rather than just printing "?"
    return (f"Stack {third_int}+{fifth_int}", f"{third_int}+{fifth_int}")


def classify_seventh(root_pc, third_pc, fifth_pc, seventh_pc):
    """
    Classify a four-note (seventh) chord: first classify the underlying
    triad, then look at the seventh's distance from the root to pick the
    seventh quality. Purely mechanical, same approach as classify_triad.
    """
    _, triad_abbrev = classify_triad(root_pc, third_pc, fifth_pc)
    seventh_int = (seventh_pc - root_pc) % 12

    table = {
        ("maj", 11): ("Major 7th", "maj7"),
        ("maj", 10): ("Dominant 7th", "dom7"),
        ("min", 10): ("Minor 7th", "min7"),
        ("min", 11): ("Minor major 7th", "minmaj7"),
        ("dim", 10): ("Half-diminished 7th", "m7b5"),
        ("dim", 9): ("Diminished 7th", "dim7"),
        ("aug", 11): ("Augmented major 7th", "augmaj7"),
        ("aug", 10): ("Augmented 7th", "aug7"),
    }
    return table.get((triad_abbrev, seventh_int), ("Other 7th", "other7"))


# Roman-numeral suffix/case conventions for each quality. Triads use the
# traditional bare uppercase/lowercase/degree-sign/plus convention;
# sevenths append the usual jazz-style suffix on top of that same
# uppercase/lowercase base.
QUALITY_SUFFIX = {
    'maj': '', 'min': '', 'dim': '\u00b0', 'aug': '+',
    'maj7': 'maj7', 'dom7': '7', 'min7': '7', 'minmaj7': '(maj7)',
    'm7b5': '\u00f87', 'dim7': '\u00b07', 'augmaj7': '+maj7', 'aug7': '+7',
    # Suspended
    'sus2': 'sus2', 'sus4': 'sus4',
    'sus2b5': 'sus2\u266d5', 'sus4b5': 'sus4\u266d5',
    'sus2#5': 'sus2\u266f5', 'sus4#5': 'sus4\u266f5',
    'sus4maj6': 'sus4(6)',
    # Add4 (open quartal voicings)
    'minadd4': 'm(add4)', 'majadd4': '(add4)',
    # Altered fifths
    'majb5': '(\u266d5)', 'min#5': 'm(\u266f5)',
    # Quartal
    'qrt': '(qrt)',
    'other': '?', 'other7': '?',
}
QUALITY_LOWERCASE = {'min', 'dim', 'min7', 'm7b5', 'dim7', 'min#5'}

# Plain-language quality words for naming a chord by what it actually
# IS (e.g. "F minor"), as opposed to the Roman-numeral/symbol notation
# used in the chart itself.
QUALITY_WORD = {
    'maj': 'major', 'min': 'minor', 'dim': 'diminished', 'aug': 'augmented',
    'maj7': 'major 7th', 'dom7': 'dominant 7th', 'min7': 'minor 7th',
    'minmaj7': 'minor-major 7th', 'm7b5': 'half-diminished 7th',
    'dim7': 'diminished 7th', 'augmaj7': 'augmented major 7th',
    'aug7': 'augmented 7th',
    'sus2': 'suspended 2nd', 'sus4': 'suspended 4th',
    'sus2b5': 'sus2 flat 5', 'sus4b5': 'sus4 flat 5',
    'sus2#5': 'sus2 aug 5', 'sus4#5': 'sus4 aug 5',
    'sus4maj6': 'sus4 add 6', 'majb5': 'major flat 5',
    'min#5': 'minor aug 5', 'qrt': 'quartal',
    'minadd4': 'minor add 4', 'majadd4': 'major add 4',
    'other': 'other', 'other7': 'other 7th',
}


def roman_numeral_for_degree(degree_1indexed, quality_abbrev):
    if degree_1indexed - 1 < len(ROMAN_BASE):
        base = ROMAN_BASE[degree_1indexed - 1]
    else:
        base = str(degree_1indexed)  # fallback for absurdly long scales
    if quality_abbrev in QUALITY_LOWERCASE:
        base = base.lower()
    return base + QUALITY_SUFFIX.get(quality_abbrev, '?')


def harmonize(scale_notes, chord_size=3):
    """
    Build a chord on every degree of the given scale (any length n >= 3),
    using degrees i, i+2, i+4[, i+6] (mod n) -- "every other scale tone"
    is the standard chord-building rule and works regardless of scale
    length, type, or whether you're building triads (chord_size=3) or
    seventh chords (chord_size=4). scale_notes: list of parse_note()
    tuples, tonic first, NOT including a repeated tonic at the top.
    """
    n = len(scale_notes)
    rows = []
    for i in range(n):
        members = [scale_notes[(i + 2 * k) % n] for k in range(chord_size)]
        pcs = [(PITCH_CLASS[m[0]] + m[1]) % 12 for m in members]

        if chord_size == 4:
            quality_name, quality_abbrev = classify_seventh(*pcs)
        else:
            quality_name, quality_abbrev = classify_triad(*pcs)

        numeral = roman_numeral_for_degree(i + 1, quality_abbrev)
        rows.append({
            'degree': i + 1,
            'numeral': numeral,
            'notes': [m[3] for m in members],
            'quality_name': quality_name,
            'quality_abbrev': quality_abbrev,
        })
    return rows


# ----------------------------------------------------------------------
# Chord function lookup: which diatonic keys (major and natural minor)
# does a given chord belong to, and what scale-degree function does it
# serve in each?
# ----------------------------------------------------------------------

LETTER_CYCLE = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
MAJOR_SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]

# Practical key tonics: up to 7 sharps or 7 flats, the conventional
# boundary before continuing the circle of fifths would call for an
# 8th accidental (at which point you'd respell enharmonically instead).
# Stored as (letter, accidental, key_signature_size) -- key_signature_size
# is how many sharps/flats THAT KEY'S SIGNATURE has (not the same as
# the tonic note's own accidental count -- e.g. F major's tonic has 0
# accidentals, but F major's key signature has 1 flat). This is what
# lets find_chord_functions prefer the simpler enharmonic spelling when
# two tonics name the same pitch (e.g. F# major's 6 sharps vs Gb
# major's 6 flats).
PRACTICAL_KEY_TONICS = [
    ('C', 0, 0), ('G', 0, 1), ('D', 0, 2), ('A', 0, 3), ('E', 0, 4), ('B', 0, 5),
    ('F', 1, 6), ('C', 1, 7),
    ('F', 0, 1), ('B', -1, 2), ('E', -1, 3), ('A', -1, 4), ('D', -1, 5),
    ('G', -1, 6), ('C', -1, 7),
]


def _scale_from_steps(tonic_letter, tonic_accidental, steps):
    """
    Build a 7-note scale spelling starting from the given tonic, one
    note per letter name in sequence (proper scale spelling, not just
    chromatic pitch classes), matching the given interval pattern.
    Returns a list of (letter, accidental) pairs.
    """
    tonic_pc = (PITCH_CLASS[tonic_letter] + tonic_accidental) % 12
    start_idx = LETTER_CYCLE.index(tonic_letter)
    notes = []
    for i in range(7):
        letter = LETTER_CYCLE[(start_idx + i) % 7]
        target_pc = (tonic_pc + steps[i]) % 12
        natural_pc = PITCH_CLASS[letter]
        acc = target_pc - natural_pc
        while acc > 6:
            acc -= 12
        while acc < -6:
            acc += 12
        notes.append((letter, acc))
    return notes


def key_display_name(letter, accidental):
    suffix = '#' * accidental if accidental > 0 else 'b' * (-accidental)
    return letter + suffix


def find_chord_functions(root_letter, root_accidental, quality_abbrev):
    """
    Search every practical major and natural-minor key for this exact
    chord (same root pitch class, same triad quality) appearing as one
    of its seven diatonic triads. Returns a list of dicts:
      {key_name, key_type ('major'/'minor'), numeral, degree}
    sorted by key type then degree, for a stable, readable order.

    Enharmonic key pairs that name the same pitches (F# major / Gb
    major, D# minor / Eb minor, etc.) are deduplicated to a single
    entry -- whichever spelling has the smaller key-signature accidental
    count wins; on an exact tie (e.g. F#/Gb major, both 6 accidentals),
    the sharp spelling is kept, matching which of the pair actually
    sees real-world use.

    This only searches TRIAD quality (maj/min/dim/aug) -- seventh-chord
    qualities aren't matched against a plain triad's diatonic role,
    since a seventh chord's function lookup would need to compare
    against the scale's harmonized sevenths instead. If the chord
    passed in is a seventh chord, its underlying triad quality is what
    gets matched (e.g. a Cmaj7 chord's root+3rd+5th is a C major triad,
    so it's matched as a major triad would be).
    """
    target_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
    # key: (tonic_pitch_class, key_type, degree) -> best candidate found
    # so far for that diatonic relationship, by accidental count.
    best_by_relationship = {}

    for tonic_letter, tonic_accidental, accidental_count in PRACTICAL_KEY_TONICS:
        tonic_pc = (PITCH_CLASS[tonic_letter] + tonic_accidental) % 12
        for key_type, steps in (('major', MAJOR_SCALE_STEPS),
                                 ('minor', NATURAL_MINOR_STEPS)):
            scale = _scale_from_steps(tonic_letter, tonic_accidental, steps)
            rows = harmonize(
                [(letter, acc, (PITCH_CLASS[letter] + acc) % 12,
                  key_display_name(letter, acc)) for letter, acc in scale],
                chord_size=3,
            )
            for row in rows:
                root_note = scale[row['degree'] - 1]
                row_root_pc = (PITCH_CLASS[root_note[0]] + root_note[1]) % 12
                if row_root_pc == target_pc and row['quality_abbrev'] == quality_abbrev:
                    rel_key = (tonic_pc, key_type, row['degree'])
                    candidate = {
                        'key_name': key_display_name(tonic_letter, tonic_accidental),
                        'key_type': key_type,
                        'numeral': row['numeral'],
                        'degree': row['degree'],
                        '_accidental_count': accidental_count,
                        '_is_sharp_side': tonic_accidental >= 0,
                    }
                    existing = best_by_relationship.get(rel_key)
                    if existing is None:
                        best_by_relationship[rel_key] = candidate
                    elif candidate['_accidental_count'] < existing['_accidental_count']:
                        best_by_relationship[rel_key] = candidate
                    elif (candidate['_accidental_count'] == existing['_accidental_count']
                          and candidate['_is_sharp_side'] and not existing['_is_sharp_side']):
                        # exact tie (e.g. F# vs Gb, both 6) -- prefer sharps
                        best_by_relationship[rel_key] = candidate

    results = list(best_by_relationship.values())
    for r in results:
        del r['_accidental_count']
        del r['_is_sharp_side']

    # Stable order: major keys first (by degree), then minor keys (by
    # degree) -- matches the example phrasing "functions as ii in F,
    # iii in Eb, vi in Bb" (major-key context first).
    results.sort(key=lambda r: (r['key_type'] != 'major', r['degree']))
    return results


def draw_chord_info_panel(chord_name, quality_name, functions, base_filename):
    """
    Render a small reference panel listing every practical major/minor
    key this chord belongs to and its Roman-numeral function there.
    functions: the list returned by find_chord_functions().
    """
    major_rows = [f for f in functions if f['key_type'] == 'major']
    minor_rows = [f for f in functions if f['key_type'] == 'minor']

    line_h = 0.42
    header_h = 0.9
    section_gap = 0.3
    n_lines = len(major_rows) + len(minor_rows)
    n_sections = sum(1 for grp in (major_rows, minor_rows) if grp)

    # Empty-result case still needs room for the two-line explanatory
    # message drawn below.
    total_h = header_h + max(n_lines, 2) * line_h + n_sections * section_gap + 0.4
    total_w = 6.5

    fig, ax = plt.subplots(figsize=(total_w, total_h), dpi=200)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    y = total_h - 0.1
    ax.text(0.15, y, chord_name, fontsize=18, fontweight='bold',
            color=CHART_LABEL_TEXT, ha='left', va='top')
    y -= 0.45
    ax.text(0.15, y, quality_name, fontsize=12, color=CHART_LABEL_TEXT,
            ha='left', va='top', style='italic')
    y -= 0.55

    if not functions:
        ax.text(0.15, y, "Not a diatonic triad of any major or natural",
                fontsize=11, color=CHART_LABEL_TEXT, ha='left', va='top')
        y -= line_h
        ax.text(0.15, y, "minor key.", fontsize=11, color=CHART_LABEL_TEXT,
                ha='left', va='top')
    else:
        if major_rows:
            ax.text(0.15, y, "MAJOR KEYS", fontsize=10, fontweight='bold',
                    color=CHART_LABEL_TEXT, ha='left', va='top')
            y -= line_h
            for r in major_rows:
                text = f"{r['numeral']}  \u2013  {r['key_name']} major"
                ax.text(0.4, y, text, fontsize=12, color=CHART_HL_STROKE,
                        ha='left', va='top', fontweight='bold')
                y -= line_h
            y -= section_gap

        if minor_rows:
            ax.text(0.15, y, "MINOR KEYS", fontsize=10, fontweight='bold',
                    color=CHART_LABEL_TEXT, ha='left', va='top')
            y -= line_h
            for r in minor_rows:
                text = f"{r['numeral']}  \u2013  {r['key_name']} minor"
                ax.text(0.4, y, text, fontsize=12, color=CHART_HL_STROKE,
                        ha='left', va='top', fontweight='bold')
                y -= line_h

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_harmonized_chart(scale_notes, scale_name, base_filename, chord_size=3):
    """
    Chromatic-column chart: each column is one semitone (not one scale
    degree), so the visual gap between highlighted boxes directly shows
    interval size -- a major third visibly spans more empty boxes than a
    minor third, even when both chords share the same "1-3-5" (or
    "1-3-5-7") scale-degree shape. Columns run 0..23, two full chromatic
    octaves from the tonic. chord_size=3 for triads, 4 for seventh
    chords.
    """
    n = len(scale_notes)
    rows = harmonize(scale_notes, chord_size=chord_size)

    root_pc = (PITCH_CLASS[scale_notes[0][0]] + scale_notes[0][1]) % 12

    # Each scale tone's chromatic offset from the tonic, 0..11.
    scale_offsets = [
        (pc - root_pc) % 12 for (_, _, pc, _) in scale_notes
    ]

    n_chromatic_cols = 24  # two full chromatic octaves
    col_w = 1.0
    row_h = 1.0
    header_h = 0.6
    subheader_h = 0.6
    label_col_w = 1.7
    quality_col_w = 3.6
    legend_h = 0.5 + 0.45 * len(set(r['quality_name'] for r in rows))

    total_w = label_col_w + n_chromatic_cols * col_w + quality_col_w
    grid_h = header_h + subheader_h + n * row_h
    total_h = grid_h + legend_h

    fig_w = max(12, n_chromatic_cols * 0.55)
    fig_h = fig_w * (total_h / total_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    def col_x(col_idx0):
        return label_col_w + col_idx0 * col_w

    top_y = total_h

    # Header row: fixed chromatic interval names (1, b2, 2, b3, 3, 4,
    # b5, 5, b6, 6, b7, 7, then 1, b9, 9, #9, b11, 11, #11, 5, b13, 13,
    # #13, 7 for the second octave) -- always the same regardless of
    # which scale is loaded; the scale only determines which of these
    # positions get highlighted below.
    header_y = top_y - header_h / 2
    for col in range(n_chromatic_cols):
        x = col_x(col)
        label = CHROMATIC_DEGREE_NAMES.get(col, str(col))
        ax.text(x + col_w / 2, header_y, label,
                ha='center', va='center', fontsize=9, color=CHART_LABEL_TEXT)

    # Subheader: scale tone name for columns that ARE a scale tone,
    # blank for chromatic passing tones not in the scale -- shows where
    # the scale itself sits within the full chromatic space.
    sub_top = top_y - header_h
    sub_y = sub_top - subheader_h / 2
    degree_at_col = {}
    for octave_rep in range(2):
        for degree_idx, offset in enumerate(scale_offsets):
            col = offset + 12 * octave_rep
            if col < n_chromatic_cols:
                degree_at_col[col] = scale_notes[degree_idx][3]  # display_name

    for col in range(n_chromatic_cols):
        x = col_x(col)
        is_scale_tone = col in degree_at_col
        fill = '#F0F0EE' if is_scale_tone else CHART_EMPTY_FILL
        rect = Rectangle((x, sub_top - subheader_h), col_w, subheader_h,
                          linewidth=0.6, edgecolor=CHART_BOX_STROKE, facecolor=fill)
        ax.add_patch(rect)
        if is_scale_tone:
            ax.text(x + col_w / 2, sub_y, degree_at_col[col],
                    ha='center', va='center', fontsize=10, color=CHART_LABEL_TEXT)

    grid_top = sub_top - subheader_h

    # Each chord row: chord tones occupy whichever scale-degree position
    # (i, i+2, i+4[, i+6]) they are, mapped to its REAL chromatic offset
    # -- this is what makes interval size visible, since two chords with
    # the same scale-degree shape can occupy different numbers of empty
    # chromatic boxes depending on the scale's actual intervals.
    for row_idx, row in enumerate(rows):
        y_top = grid_top - row_idx * row_h
        y_center = y_top - row_h / 2

        ax.text(label_col_w / 2, y_center, row['numeral'],
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=CHART_LABEL_TEXT)

        degree = row['degree']  # 1-indexed root degree
        member_degree_indices = [(degree - 1 + 2 * k) for k in range(chord_size)]
        for member_idx, note_name in zip(member_degree_indices, row['notes']):
            scale_pos = member_idx % n
            octave_reps_needed = member_idx // n
            col = scale_offsets[scale_pos] + 12 * octave_reps_needed
            x = col_x(col)
            rect = Rectangle((x, y_top - row_h), col_w, row_h,
                              linewidth=1.1, edgecolor=CHART_HL_STROKE,
                              facecolor=CHART_HL_FILL)
            ax.add_patch(rect)
            ax.text(x + col_w / 2, y_center, note_name,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=CHART_HL_TEXT)

        qx = label_col_w + n_chromatic_cols * col_w + 0.15
        ax.text(qx, y_center, row['quality_name'].upper(),
                ha='left', va='center', fontsize=11, color=CHART_LABEL_TEXT)

    # Faint grid lines for every chromatic cell in the chord rows
    for col in range(n_chromatic_cols):
        x = col_x(col)
        for row_idx in range(n):
            y_top = grid_top - row_idx * row_h
            rect = Rectangle((x, y_top - row_h), col_w, row_h,
                              linewidth=0.5, edgecolor=CHART_BOX_STROKE,
                              facecolor='none', zorder=0)
            ax.add_patch(rect)

    # Legend block at the bottom: chords grouped by quality
    legend_top = grid_top - n * row_h
    by_quality = {}
    for row in rows:
        by_quality.setdefault(row['quality_name'], []).append(row['numeral'])

    legend_y = legend_top - 0.55
    for quality_name, numerals in by_quality.items():
        label = f"{quality_name.upper()}S: " + ", ".join(numerals)
        ax.text(0.1, legend_y, label, ha='left', va='center',
                fontsize=11, fontweight='bold', color=CHART_LABEL_TEXT)
        legend_y -= 0.42

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect('equal')
    ax.axis('off')

    chord_word = "7TH CHORDS" if chord_size == 4 else "TRIADS"
    ax.text(0.1, top_y + 0.2, f"{scale_name.upper()} \u2014 HARMONIZED {chord_word}",
            ha='left', va='bottom', fontsize=15, fontweight='bold',
            color=CHART_LABEL_TEXT)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


# ----------------------------------------------------------------------
# LaTeX wrapper - New page per chord/scale
# ----------------------------------------------------------------------

def latex_escape(text):
    LATEX_SPECIAL_CHARS = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    text = text.replace('\\', '\x00')
    for ch, escaped in LATEX_SPECIAL_CHARS.items():
        if ch == '\\': continue
        text = text.replace(ch, escaped)
    text = text.replace('\x00', r'\textbackslash{}')
    return text


# Guitar chord boxes are roughly square, unlike the keyboard and
# full-neck fretboard images which are wide. Emitting them one per row
# at nearly full text width made them enormous and pushed a set of five
# CAGED shapes across three pages, so they are tiled instead.
CHORD_BOX_MARKERS = ("-caged-", "-jazz-")
GUITAR_BOX_COLUMNS = 2
GUITAR_BOX_WIDTH = 0.46      # fraction of \textwidth per tile


def _is_chord_box(image_base):
    return any(marker in image_base for marker in CHORD_BOX_MARKERS)


def _chord_box_grid(pending):
    """
    Lay out buffered chord-box images side by side, GUITAR_BOX_COLUMNS
    to a row. Each tile is a minipage so its caption stays with it, and
    the row is centred so a trailing odd tile does not hang left.
    """
    out = []
    for start in range(0, len(pending), GUITAR_BOX_COLUMNS):
        row = pending[start:start + GUITAR_BOX_COLUMNS]
        out.append(r"\begin{center}")
        tiles = []
        for label, base in row:
            tiles.append(
                rf"\begin{{minipage}}[t]{{{GUITAR_BOX_WIDTH}\textwidth}}"
                r"\centering "
                rf"\textbf{{\small {latex_escape(label)}}}\\[0.3em]"
                rf"\includegraphics[width=0.97\linewidth]{{{base}.pdf}}"
                r"\end{minipage}"
            )
        out.append(r"\hfill".join(tiles))
        out.append(r"\end{center}")
        out.append(r"\medskip")
        out.append("")
    return out


def build_lytex_wrapper(doc_title, entries_data):
    """entries_data = list of (entry_name, ly_filename, keyboard_images)"""
    parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{setspace}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{pdflscape}",
        r"\setstretch{1.15}",
        rf"\title{{{latex_escape(doc_title)}}}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{latex_escape(doc_title)}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\begin{document}",
        "",
    ]

    for entry_name, ly_filename, keyboard_images in entries_data:
        with open(ly_filename) as f:
            ly_source = f.read()

        size_start = ly_source.find("#(set-global-staff-size")
        score_block = ly_source[size_start:] if size_start != -1 else ly_source
        score_block = re.sub(r'^\s*title\s*=\s*".*"\s*$', '', score_block, flags=re.MULTILINE)

        parts.extend([
                    rf"\section*{{{latex_escape(entry_name)}}}",
                    "",
                    r"\begin{center}",
                    r"\begin{lilypond}",
                    r'\version "2.24.0"',
                    score_block.strip(),
                    r"\end{lilypond}",
                    r"\end{center}",
                    "",
                    r"\bigskip",
                ])

        # Chord boxes are buffered so consecutive ones can be tiled;
        # anything else flushes the buffer first to keep the original
        # order on the page.
        pending_boxes = []

        for inv_label, image_base in keyboard_images:
            is_harmonized_chart = image_base.endswith("-harmonized") or \
                image_base.endswith("-harmonized-7ths")

            if _is_chord_box(image_base) and not is_harmonized_chart:
                pending_boxes.append((inv_label, image_base))
                continue

            if pending_boxes:
                parts.extend(_chord_box_grid(pending_boxes))
                pending_boxes = []

            if is_harmonized_chart:
                # Harmonized chart gets its own landscape page so the
                # wide horizontal layout can breathe. pdflscape rotates
                # the physical page in the PDF viewer (not just the
                # content), making it readable without tilting your head.
                parts.extend([
                    r"\clearpage",
                    r"\begin{landscape}",
                    r"\begin{center}",
                    r"\vspace*{\fill}",
                    rf"\textbf{{\large {latex_escape(inv_label)}}}\\[0.6em]",
                    rf"\includegraphics[width=0.99\linewidth]{{{image_base}.pdf}}",
                    r"\vspace*{\fill}",
                    r"\end{center}",
                    r"\end{landscape}",
                    r"\clearpage",
                    "",
                ])
            else:
                parts.extend([
                    r"\begin{center}",
                    r"\begin{minipage}{0.92\textwidth}",
                    r"\centering",
                    rf"\textbf{{{latex_escape(inv_label)}}}\\[0.4em]",
                    rf"\includegraphics[width=0.88\textwidth]{{{image_base}.pdf}}",
                    r"\end{minipage}",
                    r"\end{center}",
                    r"\medskip",
                    "",
                ])

        if pending_boxes:
            parts.extend(_chord_box_grid(pending_boxes))
            pending_boxes = []

        parts.append(r"\newpage")   # New page for next entry

    parts.append(r"\end{document}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Song sheet -- compact grid of chord diagrams, no notation staves
# ----------------------------------------------------------------------

SONG_SHEET_COLS = 3   # chord columns per row; 3 fits comfortably on
                       # US Letter with 1in margins at the default
                       # diagram widths


def draw_blank_keyboard(base_filename, n_octaves=2, show_letter_names=True):
    """
    Draw a blank keyboard diagram with no highlighted keys.

    n_octaves: how many octaves to show (default 2).
    show_letter_names: if True, draw letter names on white keys the same
        way the regular keyboard does when no keys are highlighted.
    """
    white_names = WHITE_LETTERS * n_octaves
    n_white = len(white_names)
    total_w = n_white * KEY_W

    fig_height_in = 3.3
    fig_w = max(fig_height_in * (total_w / WHITE_H) * 0.78, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_height_in), dpi=200)

    for i, name in enumerate(white_names):
        x = i * KEY_W
        rect = FancyBboxPatch((x, 0), KEY_W, WHITE_H,
                              boxstyle="round,pad=0,rounding_size=0.06",
                              linewidth=1.2, edgecolor=WHITE_STROKE,
                              facecolor=WHITE_FILL, zorder=1)
        ax.add_patch(rect)
        if show_letter_names:
            ax.text(x + KEY_W / 2, 0.4, name, ha='center', va='center',
                    fontsize=19, color=LABEL_TEXT, fontweight='normal', zorder=3)

    for oct_idx in range(n_octaves):
        octave_x = oct_idx * 7 * KEY_W
        for boundary, _pc in BLACK_KEYS:
            x = octave_x + boundary * KEY_W - BLACK_W / 2
            rect = FancyBboxPatch((x, WHITE_H - BLACK_H), BLACK_W, BLACK_H,
                                  boxstyle="round,pad=0,rounding_size=0.04",
                                  linewidth=0.8, edgecolor="#222",
                                  facecolor="#222", zorder=2)
            ax.add_patch(rect)

    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-0.1, WHITE_H + 0.4)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(pad=0.15)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_blank_chord_box(base_filename, n_frets=4, label=None):
    """
    Draw a blank 4-fret chord diagram box with no notes marked.
    """
    string_spacing = 1.0
    fret_spacing   = 1.0
    margin_top    = 1.0
    margin_bottom = 0.6
    margin_left   = 0.8
    margin_right  = 0.5

    board_w = (FRETBOARD_N_STRINGS - 1) * string_spacing
    board_h = n_frets * fret_spacing

    fig_w = (board_w + margin_left + margin_right) * 1.1
    fig_h = (board_h + margin_top + margin_bottom) * 1.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    def sx(s): return (FRETBOARD_N_STRINGS - s) * string_spacing + margin_left
    def fy(f): return margin_top + f * fret_spacing

    # No nut bar. This drawer's y axis runs upward (unlike
    # draw_fretboard_diagram, which inverts it), so the nut was landing
    # at the bottom of the box instead of the top. A blank box is not
    # anchored to a fret position anyway, so it is simply left off and
    # the f=0 fret line closes the box.
    # Fret lines
    for f in range(n_frets + 1):
        ax.plot([margin_left, margin_left + board_w], [fy(f), fy(f)],
                color=FRETBOARD_FRET_LINE, linewidth=1.5)
    # Strings
    for s in range(1, FRETBOARD_N_STRINGS + 1):
        ax.plot([sx(s), sx(s)], [margin_top, margin_top + board_h],
                color=FRETBOARD_STRING_COLOR, linewidth=1.2)

    if label:
        ax.text((margin_left + margin_left + board_w) / 2,
                margin_top + board_h + 0.35,
                label, ha='center', va='bottom',
                fontsize=10, color=FRETBOARD_LABEL_TEXT, fontweight='bold')

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(pad=0.1)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_blank_full_fretboard(base_filename, label=None):
    """
    Draw a blank full-neck fretboard (12 frets, 6 strings) with fret
    markers but no note dots.
    """
    string_spacing = 1.0
    fret_spacing   = 0.85
    margin_top    = 0.5
    margin_bottom = 0.5
    margin_left   = 0.75
    margin_right  = 0.5

    board_w = FRETBOARD_FULL_N_FRETS * fret_spacing
    board_h = (FRETBOARD_N_STRINGS - 1) * string_spacing

    fig_w = (board_w + margin_left + margin_right) * 1.05
    fig_h = (board_h + margin_top + margin_bottom) * 1.05
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    def fret_x(f): return margin_left + f * fret_spacing
    def string_y(s): return margin_top + (s - 1) * string_spacing

    ax.add_patch(Rectangle((fret_x(0) - 0.04, margin_top - 0.15), 0.08,
                            board_h + 0.3,
                            facecolor=FRETBOARD_NUT_COLOR, edgecolor='none'))
    for f in range(1, FRETBOARD_FULL_N_FRETS + 1):
        ax.plot([fret_x(f), fret_x(f)],
                [margin_top - 0.15, margin_top + board_h + 0.15],
                color=FRETBOARD_FRET_LINE, linewidth=1.0)
    for s in range(1, FRETBOARD_N_STRINGS + 1):
        y = string_y(s)
        ax.plot([fret_x(0), fret_x(FRETBOARD_FULL_N_FRETS)], [y, y],
                color=FRETBOARD_STRING_COLOR, linewidth=1.2)

    for f in [3, 5, 7, 9, 12]:
        x = (fret_x(f) + fret_x(f - 1)) / 2
        ax.text(x, margin_top + board_h + 0.35, str(f),
                ha='center', va='top', fontsize=8,
                color=FRETBOARD_LABEL_TEXT)

    if label:
        ax.text((fret_x(0) + fret_x(FRETBOARD_FULL_N_FRETS)) / 2, -0.35,
                label, ha='center', va='bottom', fontsize=13,
                fontweight='bold', color=FRETBOARD_LABEL_TEXT)

    ax.set_xlim(margin_left - 0.55, board_w + margin_left + margin_right)
    ax.set_ylim(margin_top + board_h + 0.6, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_blank_harmonized_chart(base_filename, n_rows=7, n_octaves=2,
                                 show_interval_names=True, label=None):
    """
    Blank counterpart to draw_harmonized_chart -- the same chromatic box
    grid with nothing filled in, so the chart can be worked out by hand.

    The filled chart derives almost everything from scale data, which is
    why there was no blank version: highlighted boxes are chord tones,
    the subheader is the scale's own note names, the left column is the
    Roman numeral and the right column the chord quality. Only the
    chromatic interval-name header (1, b2, 2, b3, ... b13, 13, #13, 7)
    is scale-independent -- it is a statement about distance from the
    tonic in semitones, not about any particular scale. So the blank
    keeps that row and empties everything else, adding write-in cells
    where the filled chart prints its numerals and quality names.

    n_rows: how many chord rows (7 for a normal seven-note scale).
    n_octaves: 1 (12 columns) or 2 (24 columns, matching the filled chart).
    show_interval_names: print the interval names across the top. Turn it
        off to make the header a write-in strip as well.
    label: heading text. A "SCALE:" write-in rule is drawn beside it either way.
    """
    n_octaves = max(1, min(2, int(n_octaves)))
    n_rows = max(1, min(12, int(n_rows)))
    n_chromatic_cols = 12 * n_octaves

    # Same cell geometry as draw_harmonized_chart so a blank printed
    # alongside a filled one reads as the same chart.
    col_w = 1.0
    row_h = 1.0
    header_h = 0.6
    subheader_h = 0.6
    label_col_w = 1.7
    quality_col_w = 3.6
    heading_band_h = 1.1   # space above the grid for the heading + rule

    total_w = label_col_w + n_chromatic_cols * col_w + quality_col_w
    grid_h = header_h + subheader_h + n_rows * row_h
    total_h = grid_h + heading_band_h

    fig_w = max(12, n_chromatic_cols * 0.55)
    fig_h = fig_w * (total_h / total_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    def col_x(col_idx0):
        return label_col_w + col_idx0 * col_w

    top_y = grid_h
    header_y = top_y - header_h / 2
    sub_top = top_y - header_h
    sub_y = sub_top - subheader_h / 2
    grid_top = sub_top - subheader_h
    quality_x = label_col_w + n_chromatic_cols * col_w

    # Header row: fixed chromatic interval names, or empty boxes to
    # write them in.
    for col in range(n_chromatic_cols):
        x = col_x(col)
        if show_interval_names:
            ax.text(x + col_w / 2, header_y,
                    CHROMATIC_DEGREE_NAMES.get(col, str(col)),
                    ha='center', va='center', fontsize=9,
                    color=CHART_LABEL_TEXT)
        else:
            ax.add_patch(Rectangle((x, top_y - header_h), col_w, header_h,
                                   linewidth=0.6, edgecolor=CHART_BOX_STROKE,
                                   facecolor=CHART_EMPTY_FILL))

    # Subheader row: where the scale's own note names go.
    for col in range(n_chromatic_cols):
        ax.add_patch(Rectangle((col_x(col), sub_top - subheader_h),
                               col_w, subheader_h,
                               linewidth=0.6, edgecolor=CHART_BOX_STROKE,
                               facecolor=CHART_EMPTY_FILL))

    # Faint column captions for the two write-in side columns.
    ax.text(label_col_w / 2, header_y, "CHORD", ha='center', va='center',
            fontsize=8, color=CHART_LABEL_TEXT)
    ax.text(label_col_w / 2, sub_y, "SCALE", ha='center', va='center',
            fontsize=8, color=CHART_LABEL_TEXT)
    ax.text(quality_x + 0.15, sub_y, "QUALITY", ha='left', va='center',
            fontsize=8, color=CHART_LABEL_TEXT)

    # Chord rows: numeral cell, empty chromatic grid, quality cell.
    for row_idx in range(n_rows):
        y_top = grid_top - row_idx * row_h

        ax.add_patch(Rectangle((0, y_top - row_h), label_col_w, row_h,
                               linewidth=0.6, edgecolor=CHART_BOX_STROKE,
                               facecolor=CHART_EMPTY_FILL))

        for col in range(n_chromatic_cols):
            ax.add_patch(Rectangle((col_x(col), y_top - row_h), col_w, row_h,
                                   linewidth=0.5, edgecolor=CHART_BOX_STROKE,
                                   facecolor=CHART_EMPTY_FILL))

        ax.add_patch(Rectangle((quality_x, y_top - row_h),
                               quality_col_w, row_h,
                               linewidth=0.6, edgecolor=CHART_BOX_STROKE,
                               facecolor=CHART_EMPTY_FILL))

    # Heading band above the grid: title on the left, a "SCALE:" rule on
    # the right so the sheet says which scale it was filled in for.
    heading = (label or "Harmonized chords").upper()
    heading_y = top_y + 0.3
    ax.text(0.1, heading_y, heading, ha='left', va='bottom', fontsize=15,
            fontweight='bold', color=CHART_LABEL_TEXT)

    scale_label_x = total_w * 0.58
    ax.text(scale_label_x, heading_y, "SCALE:", ha='left', va='bottom',
            fontsize=12, color=CHART_LABEL_TEXT)
    ax.plot([scale_label_x + 2.2, total_w], [heading_y, heading_y],
            color=CHART_BOX_STROKE, linewidth=0.9)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def build_blank_staves_lytex(n_systems=8, bars_per_system=4,
                               grand_staff=False, time_sig=None,
                               staff_size=20, title=None):
    """
    Build a lilypond-book snippet producing blank staves. Each system
    has bars_per_system empty bars separated by bar lines, and \\break
    forces a new system after each group.

    A single staff (or PianoStaff) context holds all the bars; \\break
    between bar groups tells LilyPond where to start each new system.
    Multiple sibling \\new Staff blocks inside \\score are NOT valid --
    they must be one music expression.
    """
    time_cmd = f"\\time {time_sig} " if time_sig else ""

    # One group = bars_per_system empty bars. Groups are joined by \break.
    group = " ".join(["s1 |"] * bars_per_system)
    all_groups = (" \\break ").join([group] * n_systems)

    if grand_staff:
        # Both staves share the same bar structure and breaks.
        music = (
            f"\\new PianoStaff <<\n"
            f"    \\new Staff {{ \\clef treble {time_cmd}{all_groups} }}\n"
            f"    \\new Staff {{ \\clef bass  {time_cmd}{all_groups} }}\n"
            f"  >>"
        )
    else:
        music = f"\\new Staff {{ \\clef treble {time_cmd}{all_groups} }}"

    if title:
        header_body = f'title = "{latex_escape(title)}"\n  tagline = ##f'
    else:
        header_body = 'tagline = ##f'

    return (
        f'\\version "2.24.0"\n'
        f'#(set-global-staff-size {staff_size})\n'
        f'\\header {{\n'
        f'  {header_body}\n'
        f'}}\n'
        f'\\score {{\n'
        f'  {music}\n'
        f'  \\layout {{\n'
        f'    ragged-last = ##t\n'
        f'    indent = 0\\mm\n'
        f'    line-width = {PAPER_LINE_WIDTH_MM}\\mm\n'
        f'  }}\n'
        f'}}'
    )


def build_blank_templates_lytex(
        title,
        want_keyboard, keyboard_octaves, keyboard_labels,
        want_chord_boxes, chord_box_cols, chord_box_rows,
        want_full_fretboard,
        want_harmony_grid, harmony_rows=7, harmony_octaves=2,
        harmony_interval_names=True, harmony_label=None,
        want_staves=False, stave_systems=8, stave_bars=4, stave_grand=False,
        stave_time=None, stave_size=20,
        work_dir="."):
    """
    Generate all requested blank template diagrams and return a .lytex
    source string that tiles them on one or more pages.

    Returns (lytex_source, all_images) where all_images is the flat
    list of (label, base) tuples for try_run_lilypond_book.

    The harmony grid is emitted last and on its own landscape page --
    it is far too wide to sit in the portrait flow with the other
    templates, exactly as the filled harmonized chart is handled in
    build_lytex_wrapper.
    """
    # The harmony grid is the only landscape template; when it is the
    # sole selection the portrait title block would strand the title on
    # its own page (\begin{landscape} forces a page break), so the title
    # is drawn into the chart itself instead.
    harmony_only = want_harmony_grid and not any([
        want_keyboard, want_chord_boxes, want_full_fretboard, want_staves])

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{pdflscape}",
        r"\usepackage{fancyhdr}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{latex_escape(title)}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\parindent=0pt",
        r"\parskip=0pt",
        r"\begin{document}",
        "",
    ]

    if title and not harmony_only:
        parts.extend([
            rf"\begin{{center}}{{\Large\textbf{{{latex_escape(title)}}}}}\end{{center}}",
            r"\medskip",
            "",
        ])

    all_images = []

    # --- Keyboard ---
    if want_keyboard:
        kb_base = os.path.join(work_dir, "blank-keyboard")
        draw_blank_keyboard(kb_base, n_octaves=keyboard_octaves,
                            show_letter_names=keyboard_labels)
        all_images.append(("Keyboard", "blank-keyboard"))
        parts.extend([
            r"\begin{center}",
            rf"\includegraphics[width=0.95\textwidth]{{blank-keyboard.pdf}}",
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # --- Full-neck fretboard ---
    if want_full_fretboard:
        fn_base = os.path.join(work_dir, "blank-full-fretboard")
        draw_blank_full_fretboard(fn_base)
        all_images.append(("Full fretboard", "blank-full-fretboard"))
        parts.extend([
            r"\begin{center}",
            rf"\includegraphics[width=0.95\textwidth]{{blank-full-fretboard.pdf}}",
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # --- Chord box grid ---
    if want_chord_boxes:
        col_w = rf"{0.95 / chord_box_cols:.4f}\textwidth"
        total_boxes = chord_box_cols * chord_box_rows
        box_images = []
        for i in range(total_boxes):
            b = os.path.join(work_dir, f"blank-chord-box-{i}")
            draw_blank_chord_box(b)
            box_images.append(f"blank-chord-box-{i}")
            all_images.append(("Chord box", f"blank-chord-box-{i}"))

        for row_i in range(chord_box_rows):
            parts.append(r"\noindent")
            parts.append(r"\begin{minipage}[t]{\textwidth}")
            parts.append(r"\centering")
            for col_i in range(chord_box_cols):
                if col_i > 0:
                    parts.append(r"\hfill")
                idx = row_i * chord_box_cols + col_i
                parts.append(rf"\begin{{minipage}}[t]{{{col_w}}}")
                parts.append(r"\centering")
                parts.append(rf"\includegraphics[width=0.95\linewidth]{{blank-chord-box-{idx}.pdf}}")
                parts.append(r"\end{minipage}")
            parts.append(r"\end{minipage}")
            parts.append(r"\medskip")
            parts.append("")

    # --- Blank staves (lilypond-book snippet) ---
    if want_staves:
        stave_snippet = build_blank_staves_lytex(
            n_systems=stave_systems,
            bars_per_system=stave_bars,
            grand_staff=stave_grand,
            time_sig=stave_time or None,
            staff_size=stave_size,
        )
        parts.extend([
            r"\begin{center}",
            r"\begin{lilypond}",
            stave_snippet,
            r"\end{lilypond}",
            r"\end{center}",
            "",
        ])

    # --- Harmonic box chart (landscape, last page) ---
    if want_harmony_grid:
        hg_base = os.path.join(work_dir, "blank-harmony-grid")
        draw_blank_harmonized_chart(
            hg_base,
            n_rows=harmony_rows,
            n_octaves=harmony_octaves,
            show_interval_names=harmony_interval_names,
            # Only when the chart is the whole document does it carry the
            # page title; otherwise the portrait title block already has it.
            label=(harmony_label or None) if harmony_only else None,
        )
        all_images.append(("Harmonic box chart", "blank-harmony-grid"))
        parts.extend([
            r"\begin{landscape}",
            r"\begin{center}",
            r"\vspace*{\fill}",
            # Constrain height as well: a tall grid (many rows, one
            # octave) otherwise overruns the landscape page and
            # spills onto a second, blank-looking page.
            r"\includegraphics[width=0.99\linewidth,height=0.90\textheight,keepaspectratio]{blank-harmony-grid.pdf}",
            r"\vspace*{\fill}",
            r"\end{center}",
            r"\end{landscape}",
            "",
        ])

    parts.append(r"\end{document}")
    return "\n".join(parts), all_images


def build_note_name_reference_snippet(clef='treble'):
    """
    Build a LilyPond \\score block showing every note from two ledger
    lines below to two ledger lines above the staff, each labeled with
    its letter name. Uses \\language "english" so note names are plain
    letters (c d e f g a b) rather than Dutch (ces, des, ...).

    The markup placement alternates -- notes on ledger lines use _\\markup
    (below the note) to avoid collisions with the ledger line itself,
    while staff notes use ^\\markup (above).

    Returns the full lilypond-book snippet text (including \\version and
    \\score) ready to embed in a \\begin{lilypond}...\\end{lilypond} block.
    """
    # Each entry: (ly_pitch, display_name, markup_above)
    # markup_above=True  -> ^\\markup (above note)
    # markup_above=False -> _\\markup (below note)
    clef_pitches = {
        'treble': [
            # Two ledger lines below treble clef
            ("c'",  'C', False),   # C4 -- middle C, ledger below
            ("d'",  'D', False),   # D4 -- just below staff
            # Staff notes
            ("e'",  'E', True),
            ("f'",  'F', True),
            ("g'",  'G', True),
            ("a'",  'A', True),
            ("b'",  'B', True),
            ("c''", 'C', True),
            ("d''", 'D', True),
            ("e''", 'E', True),
            # Two ledger lines above treble clef
            ("f''", 'F', True),
            ("g''", 'G', True),
        ],
        'bass': [
            # Two ledger lines below bass clef
            ("e,",  'E', False),
            ("f,",  'F', False),
            # Staff notes
            ("g,",  'G', True),
            ("a,",  'A', True),
            ("b,",  'B', True),
            ("c",   'C', True),
            ("d",   'D', True),
            ("e",   'E', True),
            ("f",   'F', True),
            ("g",   'G', True),
            # Two ledger lines above bass clef
            ("a",   'A', True),
            ("b",   'B', True),
        ],
        'alto': [
            # Two ledger lines below alto clef (C clef, middle line = C4)
            ("f,",  'F', False),
            ("g,",  'G', False),
            # Staff notes
            ("a,",  'A', True),
            ("b,",  'B', True),
            ("c'",  'C', True),
            ("d'",  'D', True),
            ("e'",  'E', True),
            ("f'",  'F', True),
            ("g'",  'G', True),
            ("a'",  'A', True),
            # Two ledger lines above alto clef
            ("b'",  'B', True),
            ("c''", 'C', True),
        ],
        'tenor': [
            # Tenor C clef (4th line = C4)
            ("d,",  'D', False),
            ("e,",  'E', False),
            # Staff notes
            ("f,",  'F', True),
            ("g,",  'G', True),
            ("a,",  'A', True),
            ("b,",  'B', True),
            ("c'",  'C', True),
            ("d'",  'D', True),
            ("e'",  'E', True),
            ("f'",  'F', True),
            # Two ledger lines above tenor clef
            ("g'",  'G', True),
            ("a'",  'A', True),
        ],
    }

    pitches = clef_pitches.get(clef, clef_pitches['treble'])

    # Build the note sequence: each note is a half note (2) with its
    # name as markup. Using half notes gives enough horizontal space for
    # the labels without the staff becoming unwieldy.
    notes = []
    for ly_pitch, name, above in pitches:
        direction = '^' if above else '_'
        notes.append(
            f'{ly_pitch}2{direction}\\markup {{ \\bold "{name}" }}'
        )

    notes_str = '\n    '.join(notes)

    return f'''\\version "2.24.0"
\\language "english"
#(set-global-staff-size 16)
\\header {{ tagline = ##f }}
\\score {{
  \\new Staff {{
    \\clef {clef}
    \\omit TimeSignature
    \\cadenzaOn
    {notes_str}
  }}
  \\layout {{
    ragged-right = ##t
    indent = 0\\mm
    line-width = {PAPER_LINE_WIDTH_MM}\\mm
  }}
}}'''


def key_signature_score_for_key(tonic_letter, tonic_accidental, mode='major'):
    """
    Build a LilyPond \\score block showing just a clef and key signature
    for a directly-specified key (e.g. 'Eb major', 'F# minor'), without
    needing to infer the key from a spelled-out scale.

    mode: 'major' or 'minor' (or any LilyPond mode keyword: 'dorian',
    'mixolydian', etc.).

    Returns the \\score block text (no \\version or \\header) ready to
    embed in a lilypond-book snippet.
    """
    ly_mode_map = {
        'major': 'major', 'minor': 'minor',
        'dorian': 'dorian', 'phrygian': 'phrygian',
        'lydian': 'lydian', 'mixolydian': 'mixolydian',
        'locrian': 'locrian',
    }
    ly_tonic = to_lilypond_pitch(tonic_letter, tonic_accidental)
    ly_mode  = ly_mode_map.get(mode.lower(), 'major')
    tonic_display = key_display_name(tonic_letter, tonic_accidental)
    mode_display  = mode.capitalize()

    return f'''\\markup {{ \\bold "Key of {tonic_display} {mode_display}" }}

\\score {{
  \\new Staff \\with {{
    \\remove "Time_signature_engraver"
  }} {{
    \\clef treble
    \\key {ly_tonic} \\{ly_mode}
    \\cadenzaOn
    s1 s1 s1 s1 s1 s1
  }}
  \\layout {{
    ragged-right = ##t
    indent = 0\\mm
    line-width = 90\\mm
  }}
}}'''


def build_song_sheet_lytex(song_title, composer, key_score_block,
                            chord_entries, cols=SONG_SHEET_COLS,
                            key_scale_images=None, chart_score_block=None):
    """
    Build a lilypond-book .lytex document for the song sheet.

    Like build_song_sheet_tex, but wraps the key-signature section in a
    \\begin{lilypond}...\\end{lilypond} block so lilypond-book engraves
    it, and includes title/composer in the LaTeX document header.

    chord_entries: list of dicts with 'name', 'guitar_images',
    'keyboard_images' (same shape as build_song_sheet_tex).

    key_score_block: LilyPond \\score text from key_signature_score_for_key,
    or None if no key is specified.
    """
    col_width = rf"{0.95 / cols:.4f}\textwidth"
    title_line    = latex_escape(song_title) if song_title else "Song Sheet"
    composer_line = latex_escape(composer)   if composer   else ""

    header_parts = [rf"\textbf{{\Large {title_line}}}"]
    if composer_line:
        header_parts.append(rf"\\ \textit{{{composer_line}}}")

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{pdflscape}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{latex_escape(song_title)}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\parindent=0pt",
        r"\parskip=0pt",
        r"\begin{document}",
        "",
        r"\begin{center}",
        "\n".join(header_parts),
        r"\end{center}",
        r"\medskip",
        "",
    ]

    # Lead-sheet chart, when the chord list carried bars. It goes first:
    # the chart is the thing being read while playing, and the diagrams
    # below it are reference. Its \score carries its own \layout, so
    # unlike the key-signature block it is emitted as-is.
    if chart_score_block:
        parts.extend([
            r"\begin{lilypond}",
            r'\version "2.24.0"',
            r'\language "english"',
            r"\header { tagline = ##f }",
            chart_score_block.strip(),
            r"\end{lilypond}",
            r"\bigskip",
            "",
        ])

    # Key signature section -- engraved by lilypond-book.
    if key_score_block:
        parts.extend([
            r"\begin{center}",
            r"\begin{lilypond}",
            r'\version "2.24.0"',
            r"#(set-global-staff-size 22)",
            r"\header { tagline = ##f }",
            key_score_block.strip(),
            r"\end{lilypond}",
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # Key scale diagrams (keyboard and/or fretboard of the key's scale)
    if key_scale_images:
        parts.append(r"\begin{center}")
        for lbl, img_base in key_scale_images:
            parts.append(rf"\textbf{{{latex_escape(lbl)}}}\\[0.3em]")
            parts.append(
                rf"\includegraphics[width=0.95\textwidth]{{{img_base}.pdf}}\\"
            )
        parts.extend([r"\end{center}", r"\medskip", ""])

    # Chord diagram grid -- plain \includegraphics (images already on disk).
    for row_start in range(0, len(chord_entries), cols):
        row = chord_entries[row_start:row_start + cols]
        while len(row) < cols:
            row.append(None)

        parts.append(r"\noindent")
        parts.append(r"\begin{minipage}[t]{\textwidth}")
        parts.append(r"\centering")

        for i, entry in enumerate(row):
            if i > 0:
                parts.append(r"\hfill")
            parts.append(rf"\begin{{minipage}}[t]{{{col_width}}}")
            parts.append(r"\centering")
            if entry is not None:
                parts.append(
                    rf"\textbf{{\large {latex_escape(entry['name'])}}}\\[0.3em]"
                )
                for _lbl, img_base in entry.get('guitar_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.60\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
                for _lbl, img_base in entry.get('keyboard_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.99\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
            parts.append(r"\end{minipage}")

        parts.append(r"\end{minipage}")
        parts.append(r"\bigskip")
        parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def build_notation_sheet_lytex(title, composer, key_tonic_letter,
                                key_tonic_accidental, key_mode,
                                clef, grand_staff,
                                treble_music, bass_music,
                                key_score_block,
                                chord_entries,
                                time_sig=None,
                                use_relative=True,
                                key_scale_images=None,
                                show_note_names=False,
                                staff_size=26):
    """
    Build a lilypond-book .lytex document for the notation/lesson sheet.

    The user's raw LilyPond music content (treble_music, and optionally
    bass_music for a grand staff) is wrapped in the correct scaffold --
    \\version, \\header, \\score, \\new Staff / \\new PianoStaff. The
    user only types the notes and rhythms; all the boilerplate is added
    here.

    After the notation staff, the page continues with:
      - a key-signature reference staff (if key_score_block is given)
      - chord keyboard and/or guitar diagrams (if chord_entries given)

    chord_entries: list of dicts with 'name', 'guitar_images',
    'keyboard_images' (same shape as the song sheet).
    """
    safe_title    = latex_escape(title)    if title    else ""
    safe_composer = latex_escape(composer) if composer else ""

    # Build the LilyPond key/clef prefix that goes inside the staff.
    if key_tonic_letter:
        ly_mode_map = {
            'major': 'major', 'minor': 'minor',
            'dorian': 'dorian', 'phrygian': 'phrygian',
            'lydian': 'lydian', 'mixolydian': 'mixolydian',
            'locrian': 'locrian',
        }
        ly_tonic = to_lilypond_pitch(key_tonic_letter, key_tonic_accidental)
        ly_mode  = ly_mode_map.get((key_mode or 'major').lower(), 'major')
        key_line = f"\\key {ly_tonic} \\{ly_mode}"
    else:
        key_line = ""

    ly_clef = clef if clef else "treble"

    # Time signature prefix -- prepended to the user's music so they
    # don't have to type \time manually. If they included one in their
    # own input it will simply override this one (LilyPond uses the
    # last \time seen). Empty string means no prefix (no default time
    # signature -- LilyPond will use 4/4 implicitly).
    time_prefix = f"\\time {time_sig}\n    " if time_sig else ""

    def wrap_relative(music, ref_pitch):
        """Wrap music in \\relative unless the user already wrote one."""
        music = music.strip()
        if music.lstrip().startswith("\\relative"):
            return music  # already has it, don't double-wrap
        return f"\\relative {ref_pitch} {{\n  {time_prefix}{music}\n}}"

    # Default reference pitches: treble/alto/tenor start from c' (middle
    # C octave); bass starts from c (the octave below middle C) since
    # bass-clef writing typically lives in that register.
    bass_clef = (ly_clef == "bass")
    treble_ref = "c'" if not bass_clef else "c"
    bass_ref   = "c"

    if use_relative:
        treble_content = wrap_relative(treble_music, treble_ref)
        bass_content   = wrap_relative(bass_music, bass_ref) if bass_music else ""
    else:
        treble_content = f"{time_prefix}{treble_music.strip()}"
        bass_content   = f"{time_prefix}{bass_music.strip()}" if bass_music else ""

    # Wrap the music in the appropriate staff context.
    if grand_staff and bass_music:
        staff_block = f"""\\new PianoStaff <<
    \\new Staff {{
      \\clef {ly_clef}
      {key_line}
      {treble_content}
    }}
    \\new Staff {{
      \\clef bass
      {key_line}
      {bass_content}
    }}
  >>"""
    else:
        staff_block = f"""\\new Staff {{
    \\clef {ly_clef}
    {key_line}
    {treble_content}
  }}"""

    notation_snippet = f"""\\version "2.24.0"
\\language "english"
#(set-global-staff-size {staff_size})
\\header {{
  tagline = ##f
}}
\\score {{
  {staff_block}
  \\layout {{
    ragged-right = ##t
    indent = 0\\mm
    line-width = {PAPER_LINE_WIDTH_MM}\\mm
  }}
}}"""

    # LaTeX document preamble.
    header_parts = []
    if safe_title:
        header_parts.append(rf"\textbf{{\Large {safe_title}}}")
    if safe_composer:
        header_parts.append(rf"\\ \textit{{{safe_composer}}}")

    all_images = [img for e in chord_entries for img in
                  e.get('guitar_images', []) + e.get('keyboard_images', [])]

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{pdflscape}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{safe_title}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\parindent=0pt",
        r"\parskip=0pt",
        r"\begin{document}",
        "",
    ]

    if header_parts:
        parts.extend([
            r"\begin{center}",
            "\n".join(header_parts),
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # Main notation staff -- engraved by lilypond-book.
    parts.extend([
        r"\begin{center}",
        r"\begin{lilypond}",
        notation_snippet,
        r"\end{lilypond}",
        r"\end{center}",
        r"\medskip",
        "",
    ])

    # Optional note-name reference diagram.
    if show_note_names:
        ref_clef = ly_clef  # matches whichever clef was selected
        clef_display = ref_clef.capitalize()
        note_name_snippet = build_note_name_reference_snippet(ref_clef)
        parts.extend([
            r"\begin{center}",
            rf"\textbf{{Notes on the staff: {clef_display} clef}}\\[0.4em]",
            r"\begin{lilypond}",
            note_name_snippet,
            r"\end{lilypond}",
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # Optional key signature reference staff.
    if key_score_block:
        parts.extend([
            r"\begin{center}",
            r"\begin{lilypond}",
            r'\version "2.24.0"',
            r"#(set-global-staff-size 18)",
            r"\header { tagline = ##f }",
            key_score_block.strip(),
            r"\end{lilypond}",
            r"\end{center}",
            r"\medskip",
            "",
        ])

    # Key scale diagrams (keyboard and/or fretboard of the key's scale)
    if key_scale_images:
        parts.append(r"\begin{center}")
        for lbl, img_base in key_scale_images:
            parts.append(rf"\textbf{{{latex_escape(lbl)}}}\\[0.3em]")
            parts.append(
                rf"\includegraphics[width=0.95\textwidth]{{{img_base}.pdf}}\\"
            )
        parts.extend([r"\end{center}", r"\medskip", ""])

    # Chord reference diagrams -- keyboard and/or guitar shapes.
    if chord_entries:
        col_width = rf"{0.95 / max(1, min(3, len(chord_entries))):.4f}\textwidth"
        cols = min(3, len(chord_entries))

        for row_start in range(0, len(chord_entries), cols):
            row = chord_entries[row_start:row_start + cols]
            while len(row) < cols:
                row.append(None)

            parts.append(r"\noindent")
            parts.append(r"\begin{minipage}[t]{\textwidth}")
            parts.append(r"\centering")

            for i, entry in enumerate(row):
                if i > 0:
                    parts.append(r"\hfill")
                parts.append(rf"\begin{{minipage}}[t]{{{col_width}}}")
                parts.append(r"\centering")
                if entry is not None:
                    parts.append(
                        rf"\textbf{{\normalsize {latex_escape(entry['name'])}}}\\[0.2em]"
                    )
                    for _lbl, img_base in entry.get('guitar_images', []):
                        parts.append(
                            rf"\includegraphics[width=0.60\linewidth]{{{img_base}.pdf}}\\[0.15em]"
                        )
                    for _lbl, img_base in entry.get('keyboard_images', []):
                        parts.append(
                            rf"\includegraphics[width=0.99\linewidth]{{{img_base}.pdf}}\\[0.15em]"
                        )
                parts.append(r"\end{minipage}")

            parts.append(r"\end{minipage}")
            parts.append(r"\medskip")
            parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts)



    """
    Build a plain LaTeX document (no lilypond-book, just pdflatex) that
    tiles chord diagrams in a compact grid.

    chord_entries: list of dicts, one per chord, each with:
      'name':            display name, e.g. 'Cmaj7'
      'keyboard_images': list of (label, image_base) from draw_keyboard
      'guitar_images':   list of (label, image_base) from draw_jazz_shapes
                         / draw_all_caged_shapes (or empty list)

    Each chord gets one column; every 'cols' chords starts a new row.
    Within a column: chord name as a bold label, then guitar diagrams
    stacked, then the keyboard diagram below.

    Returns the full .tex source as a string.
    """
    col_width = rf"{0.95 / cols:.4f}\textwidth"

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{fancyhdr}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{latex_escape(song_title)}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\parindent=0pt",
        r"\parskip=0pt",
        r"\begin{document}",
        "",
        rf"\begin{{center}}{{\Large\textbf{{{latex_escape(song_title)}}}}}\end{{center}}",
        r"\medskip",
        "",
    ]

    # Key scale diagrams (keyboard and/or fretboard of the key's scale)
    if key_scale_images:
        parts.append(r"\begin{center}")
        for lbl, img_base in key_scale_images:
            parts.append(rf"\textbf{{{latex_escape(lbl)}}}\\[0.3em]")
            parts.append(
                rf"\includegraphics[width=0.95\textwidth]{{{img_base}.pdf}}\\"
            )
        parts.extend([r"\end{center}", r"\medskip", ""])

    # Chunk entries into rows of 'cols' chords
    for row_start in range(0, len(chord_entries), cols):
        row = chord_entries[row_start:row_start + cols]
        # Pad the last row with empty slots so the column widths stay
        # consistent (LaTeX minipage rows need all cells to be present).
        while len(row) < cols:
            row.append(None)

        parts.append(r"\noindent")
        parts.append(r"\begin{minipage}[t]{\textwidth}")
        parts.append(r"\centering")

        for i, entry in enumerate(row):
            if i > 0:
                parts.append(r"\hfill")
            parts.append(rf"\begin{{minipage}}[t]{{{col_width}}}")
            parts.append(r"\centering")
            if entry is not None:
                # Chord name header
                parts.append(
                    rf"\textbf{{\large {latex_escape(entry['name'])}}}\\[0.3em]"
                )
                # Guitar diagrams (jazz shapes and/or CAGED)
                for _lbl, img_base in entry.get('guitar_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.60\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
                # Keyboard diagram(s)
                for _lbl, img_base in entry.get('keyboard_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.99\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
            parts.append(r"\end{minipage}")

        parts.append(r"\end{minipage}")
        parts.append(r"\bigskip")
        parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def try_run_pdflatex(tex_filename):
    """
    Compile a plain .tex file with pdflatex (no lilypond-book step).
    Used for the song sheet, which contains only embedded PDF images
    and needs no music engraving.

    Returns (success: bool, message: str, pdf_path_or_None).
    """
    if not (shutil.which("pdflatex") or os.path.isfile("/usr/bin/pdflatex")):
        return False, "pdflatex not found on PATH.", None

    try:
        pdflatex_bin = shutil.which("pdflatex") or "/usr/bin/pdflatex"
        subprocess.run(
            [pdflatex_bin, "-interaction=nonstopmode", tex_filename],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        return False, f"pdflatex error:\n{e.stdout.strip()[-1500:]}", None

    pdf_path = tex_filename.replace(".tex", ".pdf")
    if not os.path.exists(pdf_path):
        return False, "pdflatex ran but no PDF was produced.", None

    return True, f"Song sheet PDF: {pdf_path}", pdf_path




def lilypond_book_path():
    """
    Resolved path to lilypond-book, or None. gunicorn may run with a
    restricted PATH, so the standard apt location is checked directly as
    well -- same fallback the build helpers use. Exposed so a caller can
    refuse a lilypond-only feature up front instead of discovering it
    after drawing everything.
    """
    return shutil.which("lilypond-book") or (
        "/usr/bin/lilypond-book"
        if os.path.isfile("/usr/bin/lilypond-book") else None
    )


def try_run_lilypond_book(lytex_filename, all_keyboard_images, out_dir="lilypond-book-out"):
    tex_filename = lytex_filename.replace(".lytex", ".tex")

    # shutil.which searches PATH; gunicorn may run with a restricted PATH
    # so also check the standard apt install location directly.
    lb_path = lilypond_book_path()

    if lb_path is None:
        image_list = "\n".join(f"    cp {base}.pdf {out_dir}/" for _, base in all_keyboard_images)
        return False, (
            "lilypond-book was not found...\n"
            f"    lilypond-book --pdf --output={out_dir} {lytex_filename}\n"
            f"{image_list}\n"
            f"    cd {out_dir} && pdflatex {tex_filename} && cd .."
        )

    try:
        subprocess.run([lb_path, "--pdf", f"--output={out_dir}", lytex_filename],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, f"lilypond-book error:\n{e.stderr.strip()[-1500:]}"

    for _, image_base in all_keyboard_images:
        src = f"{image_base}.pdf"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, src))

    if not (shutil.which("pdflatex") or os.path.isfile("/usr/bin/pdflatex")):
        return False, f"pdflatex not found. Run: cd {out_dir} && pdflatex {tex_filename}"

    try:
        pdflatex_bin = shutil.which("pdflatex") or "/usr/bin/pdflatex"
        subprocess.run([pdflatex_bin, "-interaction=nonstopmode", tex_filename],
                       check=True, capture_output=True, text=True, cwd=out_dir)
    except subprocess.CalledProcessError as e:
        return False, f"pdflatex error:\n{e.stdout.strip()[-1500:]}"

    final_pdf = tex_filename.replace(".tex", ".pdf")
    return True, f"Final combined PDF written to ./{out_dir}/{final_pdf}"


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------

def build_scale_from_key(tonic_letter, tonic_accidental, mode='major'):
    """
    Build a parsed scale (list of parse_note() tuples) from a tonic
    note and mode name, using the same interval patterns as CHURCH_MODES.

    Returns a list of 7 parse_note()-style tuples (letter, accidental,
    pc, display_name), one per scale degree, root first. No trailing
    tonic repeat -- callers add that if needed.

    The scale degrees are spelled using consecutive letter names starting
    from the tonic, so the result is always unambiguously notated (e.g.
    C major = C D E F G A B, not C D E F G Ab Bb).
    """
    mode_key = {
        'major': 'ionian', 'minor': 'aeolian',
    }.get(mode.lower(), mode.lower())
    intervals = CHURCH_MODES.get(mode_key, CHURCH_MODES['ionian'])

    tonic_pc = (PITCH_CLASS[tonic_letter] + tonic_accidental) % 12
    root_idx  = LETTERS.index(tonic_letter)
    result    = []

    for degree_idx, semitones in enumerate(intervals):
        letter = LETTERS[(root_idx + degree_idx) % 7]
        pc     = (tonic_pc + semitones) % 12
        # Compute the accidental: how many semitones does this letter
        # naturally sit away from the target pc?
        natural_pc = PITCH_CLASS[letter]
        diff       = (pc - natural_pc) % 12
        # Normalise to the range [-2, 2] (double flat to double sharp)
        if diff > 6:
            diff -= 12
        accidental    = diff
        display_name  = key_display_name(letter, accidental)
        result.append((letter, accidental, pc, display_name))

    return result


def draw_key_scale_diagrams(tonic_letter, tonic_accidental, mode,
                             slug_prefix, want_keyboard=True,
                             want_fretboard=False):
    """
    Generate keyboard and/or full-fretboard scale diagrams for a given
    key/mode, using the same functions as the main scale generator.

    Returns a list of (label, image_base) tuples for both pipelines.
    """
    mode_key = {
        'major': 'ionian', 'minor': 'aeolian',
    }.get(mode.lower(), mode.lower())
    tonic_display = key_display_name(tonic_letter, tonic_accidental)
    mode_display  = MODE_DISPLAY_NAME.get(mode_key, mode.capitalize())
    scale_name    = f"{tonic_display} {mode_display}"

    parsed = build_scale_from_key(tonic_letter, tonic_accidental, mode)
    images = []

    if want_keyboard:
        notes = assign_octaves(parsed, start_octave=4, mode='scale')
        degree_labels = degree_labels_for_voicing(
            notes, tonic_letter, tonic_accidental, extension_pcs=set()
        )
        kb_base = f"{slug_prefix}-key-scale-kb"
        draw_keyboard(notes, degree_labels, kb_base)
        images.append((f"Scale / Key of {tonic_display} {mode_display}", kb_base))

    if want_fretboard:
        fret_images = draw_full_fretboard_scale(parsed, scale_name,
                                                f"{slug_prefix}-key-scale")
        images.extend(fret_images)

    return images


def build_song_sheet_tex(song_title, chord_entries, cols=SONG_SHEET_COLS,
                          key_scale_images=None):
    """
    Build a plain LaTeX document (no lilypond-book, just pdflatex) that
    tiles chord diagrams in a compact grid. Used as a fallback when
    lilypond-book is unavailable (no key-signature staff in this path).

    chord_entries: list of dicts with 'name', 'guitar_images',
    'keyboard_images' (same shape as build_song_sheet_lytex).
    """
    col_width = rf"{0.95 / cols:.4f}\textwidth"

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{pdflscape}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lfoot{{\small\textit{{{latex_escape(song_title)}}}}}",
        r"\rfoot{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\parindent=0pt",
        r"\parskip=0pt",
        r"\begin{document}",
        "",
        rf"\begin{{center}}{{\Large\textbf{{{latex_escape(song_title)}}}}}\end{{center}}",
        r"\medskip",
        "",
    ]

    for row_start in range(0, len(chord_entries), cols):
        row = chord_entries[row_start:row_start + cols]
        while len(row) < cols:
            row.append(None)

        parts.append(r"\noindent")
        parts.append(r"\begin{minipage}[t]{\textwidth}")
        parts.append(r"\centering")

        for i, entry in enumerate(row):
            if i > 0:
                parts.append(r"\hfill")
            parts.append(rf"\begin{{minipage}}[t]{{{col_width}}}")
            parts.append(r"\centering")
            if entry is not None:
                parts.append(
                    rf"\textbf{{\large {latex_escape(entry['name'])}}}\\[0.3em]"
                )
                for _lbl, img_base in entry.get('guitar_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.60\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
                for _lbl, img_base in entry.get('keyboard_images', []):
                    parts.append(
                        rf"\includegraphics[width=0.99\linewidth]{{{img_base}.pdf}}\\[0.2em]"
                    )
            parts.append(r"\end{minipage}")

        parts.append(r"\end{minipage}")
        parts.append(r"\bigskip")
        parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def draw_keyboard_inversions(parsed, name, slug, inversion_indices=None,
                             rootless=False):
    """
    Draw one keyboard diagram per requested inversion of a chord.

    parsed: list of parse_note() tuples, root first.
    name: display name for the chord (used in image labels).
    slug: filename slug prefix.
    inversion_indices: list of 0-based inversion indices to render,
        e.g. [0] for root only, [0, 1, 2] for root + two inversions.
        None or [] defaults to [0] (root position only).
    rootless: drop the root from the voicing, the left hand or the bass
        player being assumed to cover it. Degrees are still measured
        from the root that was removed, so the labels stay 3/b7/9/13.

    Returns a list of (label, image_base) tuples.
    """
    if not inversion_indices:
        inversion_indices = [0]

    # The root is removed before inversions are generated, so the
    # inversions are rotations of what actually sounds -- otherwise
    # 'first inversion' of a rootless voicing would still be counted
    # from a note that is not being played.
    voiced = parsed
    if rootless:
        root_reference_pc = (PITCH_CLASS[parsed[0][0]] + parsed[0][1]) % 12
        voiced = [
            note for note in parsed[1:]
            if (PITCH_CLASS[note[0]] + note[1]) % 12 != root_reference_pc
        ]
        if not voiced:
            return []

    n_notes = len(voiced)
    # Clamp indices to the valid range -- a 4-note chord has inversions
    # 0-3; asking for inversion 5 on a triad silently caps to 2.
    indices = [max(0, min(k, n_notes - 1)) for k in inversion_indices]
    # Deduplicate while preserving order.
    seen = set()
    indices = [k for k in indices if not (k in seen or seen.add(k))]

    root_letter     = parsed[0][0]
    root_accidental = parsed[0][1]
    root_pc         = (PITCH_CLASS[root_letter] + root_accidental) % 12
    root_ref_abs    = 4 * 12 + root_pc

    # Compute extension_pcs once from root-position voicing so that
    # every inversion's labels reflect the same harmonic register intent.
    root_pos_voicing = assign_octaves(voiced, start_octave=4, mode='chord')
    ext_pcs = {
        n["pc"] for n in root_pos_voicing
        if (n["octave"] * 12 + n["pc"]) - root_ref_abs > 12
    }

    all_inversions = generate_inversions(voiced, max(indices) + 1)

    images = []
    # Each inversion sizes its own keyboard. Forcing a common width used
    # to keep them comparable, but it means the widest inversion sets
    # the size for all of them -- for Gb7#11 one inversion needs 22
    # white keys, so the root position was drawn at 22 too when 15 would
    # do. LaTeX scales every image to the same column width anyway, so a
    # narrower window simply prints with bigger keys. To go back to
    # uniform sizing, pass n_white_keys=max(keyboard_white_span(inv)[1]
    # for inv in all_inversions) to draw_keyboard below.

    for k in indices:
        inv = all_inversions[k]
        inv_label_short = inversion_name(k)
        labels = degree_labels_for_voicing(
            inv, root_letter, root_accidental, extension_pcs=ext_pcs
        )
        img_label = f"{name} \u2013 {inv_label_short}"
        img_base  = f"{slug}-kb-inv{k}"
        draw_keyboard(inv, labels, img_base)
        images.append((img_label, img_base))

    return images


def slugify(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # '#' (sharp) would otherwise just get silently stripped by the
    # punctuation regex below -- since 'b' (flat) is already a letter
    # and survives on its own, that asymmetry meant "F# major" and
    # "F major" both slugified to "f-major". Spell sharps out so they
    # survive too, and so two different keys never collide on disk.
    text = text.replace('#', 'sharp')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[\s_-]+', '-', text)
    return text or "untitled"


def read_notes_prompt(prompt_text):
    raw = input(prompt_text).strip()
    tokens = raw.split()
    if not tokens:
        print("No notes entered.")
        sys.exit(1)
    try:
        return [parse_note(t) for t in tokens]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# Guitar fretboard diagrams: the CAGED system
# ----------------------------------------------------------------------
#
# Five movable major shapes (named after the open chords they're
# derived from: C, A, G, E, D) and five corresponding minor shapes,
# each verified against known music theory before being hardcoded here
# -- rather than searching for "a" playable fingering algorithmically,
# this reproduces the specific, recognizable shapes guitarists actually
# learn, transposed to whatever root the chord needs.

GUITAR_OPEN_STRING_PC = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}  # E A D G B E

CAGED_ORDER = ['C', 'A', 'G', 'E', 'D']

CAGED_SHAPES_MAJOR = {
    'C': {6: 'x', 5: 3, 4: 2, 3: 0, 2: 1, 1: 0},
    'A': {6: 'x', 5: 0, 4: 2, 3: 2, 2: 2, 1: 0},
    'G': {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3},
    'E': {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0},
    'D': {6: 'x', 5: 'x', 4: 0, 3: 2, 2: 3, 1: 2},
}
CAGED_SHAPE_ROOT_PC = {'C': 0, 'A': 9, 'G': 7, 'E': 4, 'D': 2}


def _guitar_pc_at(string, fret):
    return (GUITAR_OPEN_STRING_PC[string] + fret) % 12


def _build_caged_minor_shape(shape_name):
    """
    Derive a minor-shape template from its major counterpart by
    flattening the major 3rd by one semitone -- the standard way
    guitarists actually play the (rarely fully-open) Cm/Gm shapes, and
    exactly how Em/Am/Dm relate to their major counterparts too.
    Returns (template, min_legal_offset) where min_legal_offset is how
    far this template must be shifted up the neck before every string
    lands on a non-negative fret (Cm/Gm's "natural" position has a
    string at fret -1, which isn't playable -- those two shapes are
    only ever used barred higher up the neck in practice).
    """
    major_shape = CAGED_SHAPES_MAJOR[shape_name]
    root_pc = CAGED_SHAPE_ROOT_PC[shape_name]
    major_third_pc = (root_pc + 4) % 12
    minor_shape = {}
    for string, fret in major_shape.items():
        if fret == 'x':
            minor_shape[string] = 'x'
            continue
        pc = _guitar_pc_at(string, fret)
        minor_shape[string] = fret - 1 if pc == major_third_pc else fret
    min_fret = min(f for f in minor_shape.values() if f != 'x')
    min_legal_offset = max(0, -min_fret)
    return minor_shape, min_legal_offset


CAGED_SHAPES_MINOR = {}
CAGED_MIN_LEGAL_OFFSET = {'maj': {n: 0 for n in CAGED_ORDER}, 'min': {}}
for _shape_name in CAGED_ORDER:
    _tmpl, _min_legal = _build_caged_minor_shape(_shape_name)
    CAGED_SHAPES_MINOR[_shape_name] = _tmpl
    CAGED_MIN_LEGAL_OFFSET['min'][_shape_name] = _min_legal

CAGED_SHAPES_BY_QUALITY = {'maj': CAGED_SHAPES_MAJOR, 'min': CAGED_SHAPES_MINOR}

# Which triad member ('1', '3', or '5') each string plays in each
# shape -- fixed by the shape's own construction and unchanged by
# transposition or by major/minor (flattening the 3rd doesn't move it
# to a different string, just changes its quality). Verified by
# computing role = (pitch_class_at(string,fret) - shape_root_pc) % 12
# for every open-position template and checking it lands on 0 (root),
# 4 (major 3rd), or 7 (fifth) -- see the working notes in this
# project's history for the full derivation table.
CAGED_SHAPE_STRING_ROLES = {
    'C': {5: '1', 4: '3', 3: '5', 2: '1', 1: '3'},
    'A': {5: '1', 4: '5', 3: '1', 2: '3', 1: '5'},
    'G': {6: '1', 5: '3', 4: '5', 3: '1', 2: '3', 1: '1'},
    'E': {6: '1', 5: '5', 4: '1', 3: '3', 2: '5', 1: '1'},
    'D': {4: '1', 3: '5', 2: '1', 1: '3'},
}


def get_caged_shape(shape_name, quality, target_root_pc):
    """
    quality: 'maj' or 'min'. Returns (fret_dict, lowest_fret_used).
    fret_dict maps string number (1-6) to a fret (int, 0 = open) or
    'x' (muted). If the shape's natural transposition would require a
    negative fret (only possible for the Cm/Gm minor shapes at their
    own root), shifts up a full octave (+12 frets) instead -- matching
    how guitarists actually use those two shapes only higher up the
    neck, never as an open chord.
    """
    template = CAGED_SHAPES_BY_QUALITY[quality][shape_name]
    shape_root_pc = CAGED_SHAPE_ROOT_PC[shape_name]
    required_offset = (target_root_pc - shape_root_pc) % 12
    min_legal = CAGED_MIN_LEGAL_OFFSET[quality][shape_name]
    offset = required_offset if required_offset >= min_legal else required_offset + 12

    result = {}
    for string, fret in template.items():
        result[string] = 'x' if fret == 'x' else fret + offset
    lowest_fret_used = min(f for f in result.values() if f != 'x')
    return result, lowest_fret_used


FRETBOARD_LABEL_TEXT = "#444441"
FRETBOARD_FRET_LINE = "#999999"
FRETBOARD_NUT_COLOR = "#2C2C2A"
FRETBOARD_STRING_COLOR = "#777771"
FRETBOARD_HL_FILL = "#D85A30"
FRETBOARD_HL_STROKE = "#993C1D"
FRETBOARD_HL_TEXT = "#FFFFFF"
FRETBOARD_MUTE_COLOR = "#B0B0AA"
FRETBOARD_OPEN_RING_COLOR = "#444441"
# Root/tonic gets its own color on the full-fretboard overview so it
# reads at a glance, the same way a root note is conventionally called
# out on printed scale/chord reference charts. Picked to sit clearly
# apart from the coral used for every other tone, while still fitting
# the same warm, muted palette as the rest of the diagrams.
FRETBOARD_ROOT_FILL = "#2C2825"
FRETBOARD_ROOT_STROKE = "#000000"
FRETBOARD_ROOT_TEXT = "#FFFFFF"

FRETBOARD_N_STRINGS = 6
FRETBOARD_MIN_FRETS_SHOWN = 4

# Chord-box annotation sizing. These were previously inline literals in
# draw_fretboard_diagram; pulled out so the position label and the muted
# -string marks can be tuned without hunting through the drawing code.
# Both read small when a page tiles several diagrams at 0.60\linewidth,
# so they are deliberately larger than the degree labels inside the dots.
FRETBOARD_FRET_LABEL_FONTSIZE = 45     # the "3fr" position marker
# The muted-string mark is drawn as two stroked lines rather than a
# text glyph. The multiplication-sign glyph it used before has thin
# strokes even at fontweight='bold', so it stayed faint next to the
# open-string rings no matter how much the point size went up; a drawn
# X gives real, tunable weight. Sizes are in diagram units, where the
# string spacing is 1.0 and an open-string ring is 0.32 across.
FRETBOARD_MUTE_HALF_SIZE = 0.20        # half the width of the X
FRETBOARD_MUTE_LINEWIDTH = 2.6
# The mark sits above the nut; a bigger X needs more clearance or it
# collides with the nut bar and the open-string rings beside it.
FRETBOARD_MUTE_OFFSET = 0.45
# Fingering dots and their degree labels. Sizes are in diagram units
# where the string spacing is 1.0, so a radius of 0.28 leaves a clear
# gap between dots on neighbouring strings. These were 0.22/9pt, which
# read fine on screen but fell apart in print once a page tiles several
# diagrams at a fraction of the column width.
FRETBOARD_DOT_RADIUS = 0.30
FRETBOARD_DOT_LABEL_FONTSIZE = 12
# Open-string rings sit above the nut and carry the same labels, so
# they scale with the dots.
FRETBOARD_OPEN_RING_RADIUS = 0.22
FRETBOARD_OPEN_RING_FONTSIZE = 10


def draw_fretboard_diagram(frets, lowest_fret_used, degree_labels_by_string,
                            shape_name, chord_name, base_filename,
                            show_title=False):
    """
    frets: dict {string_number(1-6): fret_int or 'x'}, string 6 = low E,
    string 1 = high E (matches get_caged_shape's output directly).
    lowest_fret_used: from get_caged_shape -- decides whether this
    renders as an open-position diagram (nut at the top) or a movable
    barre position (plain top line + "Nfr" label).
    degree_labels_by_string: dict {string_number: label_str}, e.g.
    '1', 'b3', '5' for each fretted/open note -- not shown for muted
    strings.
    shape_name: one of 'C', 'A', 'G', 'E', 'D'.
    """
    is_open_position = lowest_fret_used == 0
    display_start_fret = 0 if is_open_position else lowest_fret_used

    highest_fret_used = max(f for f in frets.values() if f != 'x')
    n_frets_needed = highest_fret_used - display_start_fret + 1
    n_frets_shown = max(FRETBOARD_MIN_FRETS_SHOWN, n_frets_needed)

    string_spacing = 1.0
    fret_spacing = 1.0
    margin_top = 1.0
    margin_bottom = 0.6
    margin_left = 0.8
    margin_right = 0.5

    board_w = (FRETBOARD_N_STRINGS - 1) * string_spacing
    board_h = n_frets_shown * fret_spacing

    fig_w = (board_w + margin_left + margin_right) * 1.1
    fig_h = (board_h + margin_top + margin_bottom) * 1.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    def string_x(string_num):
        return (FRETBOARD_N_STRINGS - string_num) * string_spacing

    def fret_y(fret_num_relative):
        return margin_top + fret_num_relative * fret_spacing

    if is_open_position:
        ax.add_patch(Rectangle((0, margin_top - 0.06), board_w, 0.12,
                                facecolor=FRETBOARD_NUT_COLOR, edgecolor='none'))
    else:
        ax.plot([0, board_w], [margin_top, margin_top],
                color=FRETBOARD_FRET_LINE, linewidth=1.2)

    for i in range(1, n_frets_shown + 1):
        y = fret_y(i)
        ax.plot([0, board_w], [y, y], color=FRETBOARD_FRET_LINE, linewidth=1.0)

    for s in range(1, FRETBOARD_N_STRINGS + 1):
        x = string_x(s)
        ax.plot([x, x], [margin_top, margin_top + board_h],
                color=FRETBOARD_STRING_COLOR, linewidth=1.2)

    # The position label names the fret the ROOT sits on, not the top of
    # the window -- that is the number a player has in mind for a shape
    # ("the Bb7 at the sixth fret"), and the two differ whenever the
    # voicing reaches below its root. The root is whichever string is
    # labelled '1'; when a shape doubles the root, the lowest-pitched
    # one wins (highest string number), and open roots are skipped so
    # that a shape with an open bass root still names its fretted
    # position. The label is then drawn beside the root's own fret row
    # rather than the first row, so the number points at the fret it
    # names.
    root_fret_candidates = [
        (s, frets[s]) for s, lbl in degree_labels_by_string.items()
        if lbl == '1' and isinstance(frets.get(s), int) and frets[s] > 0
    ]
    if root_fret_candidates:
        label_fret = max(root_fret_candidates)[1]
    else:
        label_fret = display_start_fret
    label_row = (label_fret - display_start_fret) + 1 if not is_open_position else 1

    # Reserve the position-label gutter on EVERY diagram, whether or not
    # a label is drawn, using an invisible widest-case spacer. These are
    # saved with bbox_inches='tight', so the crop follows the drawn
    # content: a diagram carrying "fr. 12" cropped wider than an
    # open-position one, and since LaTeX scales them all to the same
    # \linewidth fraction, the wider image came out with a smaller
    # board. Same-width crops keep every board the same size. alpha=0
    # rather than set_visible(False) -- an invisible artist is dropped
    # from the bbox, a transparent one still counts.
    ax.text(-0.55, fret_y(0.5), "fr. 12",
            ha='right', va='center',
            fontsize=FRETBOARD_FRET_LABEL_FONTSIZE,
            fontweight='bold', alpha=0)

    if not is_open_position:
        ax.text(-0.55, fret_y(label_row - 0.5), f"fr. {label_fret}",
                ha='right', va='center',
                fontsize=FRETBOARD_FRET_LABEL_FONTSIZE,
                color=FRETBOARD_LABEL_TEXT,
                fontweight='bold')

    for s in range(1, FRETBOARD_N_STRINGS + 1):
        x = string_x(s)
        fret = frets[s]
        if fret == 'x':
            cy = margin_top - FRETBOARD_MUTE_OFFSET
            h = FRETBOARD_MUTE_HALF_SIZE
            for dx in (h, -h):
                ax.plot([x - dx, x + dx], [cy - h, cy + h],
                        color=FRETBOARD_MUTE_COLOR,
                        linewidth=FRETBOARD_MUTE_LINEWIDTH,
                        solid_capstyle='round', zorder=3)
            continue
        if fret == 0:
            circ = Circle((x, margin_top - 0.32), FRETBOARD_OPEN_RING_RADIUS,
                          facecolor='none',
                          edgecolor=FRETBOARD_OPEN_RING_COLOR, linewidth=1.5)
            ax.add_patch(circ)
            label = degree_labels_by_string.get(s, '')
            if label:
                ax.text(x, margin_top - 0.32, label, ha='center', va='center',
                        fontsize=FRETBOARD_OPEN_RING_FONTSIZE,
                        color=FRETBOARD_OPEN_RING_COLOR, fontweight='bold')
            continue
        # Fretted note: filled dot at the midpoint of its fret cell.
        # When in open position, fret N sits in cell N directly. When
        # showing a barred/non-open window (display_start_fret > 0), a
        # note exactly AT display_start_fret must land in the FIRST
        # visible cell, not above the diagram -- without the +1 here,
        # a barre at the window's own starting fret computes a
        # relative position of 0, which sits ABOVE the top line (where
        # open-string rings are drawn), not inside the grid.
        if is_open_position:
            relative_fret = fret
        else:
            relative_fret = (fret - display_start_fret) + 1
        y = fret_y(relative_fret - 0.5)
        dot = Circle((x, y), FRETBOARD_DOT_RADIUS, facecolor=FRETBOARD_HL_FILL,
                      edgecolor=FRETBOARD_HL_STROKE, linewidth=1.3, zorder=3)
        ax.add_patch(dot)
        label = degree_labels_by_string.get(s, '')
        if label:
            # Three-character labels (b13, #11, bb7) are the tight case;
            # shrink just those rather than sizing every dot for them.
            fs = FRETBOARD_DOT_LABEL_FONTSIZE
            if len(label) >= 3:
                fs *= 0.78
            ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                    color=FRETBOARD_HL_TEXT, fontweight='bold', zorder=4)

    # Matching spacer for the top-right corner: a muted string 1 puts an
    # X half a cell past the board edge and above the nut, so without
    # this the crop would depend on whether the voicing happens to mute
    # the top string.
    ax.plot([board_w + FRETBOARD_MUTE_HALF_SIZE],
            [margin_top - FRETBOARD_MUTE_OFFSET - FRETBOARD_MUTE_HALF_SIZE],
            marker='.', alpha=0)

    ax.set_xlim(-margin_left, board_w + margin_right)
    ax.set_ylim(margin_top + board_h + margin_bottom, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')

    # The title that used to sit above each diagram ("Cmaj7 - str.6,
    # fr.8") is not drawn any more: the LaTeX builders already caption
    # each image, and at the size these tile onto a page it was too
    # small to read. Pass show_title=True to bring it back.
    if show_title:
        title = (f"{chord_name} \u2013 {shape_name}-shape"
                 if len(shape_name) == 1
                 else f"{chord_name} \u2013 {shape_name}")
        ax.text(board_w / 2, -0.45, title, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=FRETBOARD_LABEL_TEXT)

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_all_caged_shapes(root_letter, root_accidental, quality_abbrev, chord_name,
                           base_slug):
    """
    Generate all 5 CAGED shapes for a chord (major or minor triad
    quality only -- CAGED is fundamentally a triad-shape system, so a
    7th chord's underlying triad quality is what's used here). Returns
    a list of (label, image_base) tuples, ready to drop straight into
    the same keyboard_images list the rest of the pipeline already
    uses.

    Degree labels (1, 3/b3, 5) come from CAGED_SHAPE_STRING_ROLES, a
    hardcoded table of which triad member (root/third/fifth) each
    string plays in each shape -- this is fixed by the shape's own
    structure and never changes under transposition, so labeling this
    way is exact. The earlier approach (reverse-engineering a note's
    letter name from its bare pitch class, then computing a degree
    from that guessed spelling) was WRONG: pitch class 1 is ambiguous
    between C# and Db, and defaulting to a flat spelling produced
    nonsense labels like "b4" for what is structurally a plain major
    3rd. Since each shape's triad role per string is already known
    from the shape's own construction, there's no spelling to guess.
    """
    if quality_abbrev not in ('maj', 'min'):
        return []  # CAGED shapes are only defined for plain triads

    root_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
    third_label = '3' if quality_abbrev == 'maj' else 'b3'
    role_label = {'1': '1', '3': third_label, '5': '5'}

    images = []
    for shape_name in CAGED_ORDER:
        frets, lowest = get_caged_shape(shape_name, quality_abbrev, root_pc)
        roles = CAGED_SHAPE_STRING_ROLES[shape_name]
        labels = {string: role_label[roles[string]] for string in roles}
        img_base = f"{base_slug}-caged-{shape_name.lower()}"
        draw_fretboard_diagram(frets, lowest, labels, shape_name, chord_name, img_base)
        images.append((f"{chord_name} \u2013 {shape_name}-shape (guitar)", img_base))
    return images


# A major or minor third voiced on the A string under a root on the low
# E string is muddy down the neck: with the root at fret r the third
# lands on fret r-1 or r-2, so a root at the 5th fret puts the third
# around C#3. Below this root fret the third is kept off string 5
# entirely, which pushes the search to voice it an octave up on one of
# the thinner strings. Above it the interval has climbed far enough to
# be worth having. Raise this to be stricter, or set it to 0 to allow
# the low third everywhere.
JAZZ_LOW_THIRD_MIN_ROOT_FRET = 7


def find_jazz_voicings(chord_pcs, root_pc, root_strings=(6, 5),
                       fret_span=4, optional_pcs=None, max_voicings=4,
                       extension_pcs=None, bass_cutoff_string=4,
                       pc_to_label=None,
                       low_third_min_root_fret=JAZZ_LOW_THIRD_MIN_ROOT_FRET,
                       include_open_pedal=False, rootless=False):
    """
    Find practical jazz voicings for a chord, with the root on a chosen
    bass string (6=low E or 5=A) and the remaining tones voiced within
    a fret_span-fret window on the higher strings.

    extension_pcs: pitch classes treated as upper-register extensions
    (9, 11, 13 etc.) -- excluded from strings at or below
    bass_cutoff_string to avoid muddy low-register voicings.

    bass_cutoff_string: strings numbered >= this are not allowed to
    voice extension_pcs. Default 4 (D string) -- extensions only on
    strings 1, 2, 3.

    include_open_pedal: also return pedal voicings -- an open string
    sounding the root under a grip further up the neck (see the second
    enumeration pass below). Off by default; these are a distinct
    playing idea rather than more of the same movable shapes, so they
    are opt-in.

    pc_to_label: dict {pc: label_string} for labeling each note.
    Computed from the user's spelling by draw_jazz_shapes so labels
    reflect the chord's actual register intent ('6' vs '13', '9' vs
    '2') rather than a generic interval table.
    """
    if optional_pcs is None:
        optional_pcs = {(root_pc + 7) % 12}
    if extension_pcs is None:
        extension_pcs = set()
    if pc_to_label is None:
        pc_to_label = {}

    # The third is identified from the labels rather than from an
    # interval count, so an oddly spelled chord still resolves it: any
    # label that is a 3 with or without an accidental.
    third_pcs = {
        pc for pc, lbl in pc_to_label.items()
        if lbl.lstrip('#b') == '3'
    }

    required_pcs = {pc for pc in chord_pcs if pc not in optional_pcs}
    required_pcs.add(root_pc)
    all_target_pcs = set(chord_pcs)

    # Rootless: the root is neither required nor allowed anywhere in the
    # shape, and the voicing is anchored on a guide tone instead. The
    # third and seventh are what define the harmony once the root is
    # gone, so they are what the bass string looks for; a chord with
    # neither falls back to anchoring on any available tone.
    anchor_pcs = {root_pc}
    if rootless:
        seventh_pcs = {
            pc for pc, lbl in pc_to_label.items()
            if lbl.lstrip('#b') in ('7', 'b7')
        }
        guide_pcs = (third_pcs | seventh_pcs) - {root_pc}
        anchor_pcs = guide_pcs or (all_target_pcs - {root_pc})
        if not anchor_pcs:
            return []
        required_pcs = required_pcs - {root_pc}
        all_target_pcs = all_target_pcs - {root_pc}
        # Nothing to double an absent root, and no root to pedal.
        include_open_pedal = False

    results = []

    for root_string in root_strings:
        root_frets_on_string = [
            f for f in range(0, 13)
            if (GUITAR_OPEN_STRING_PC[root_string] + f) % 12 in anchor_pcs
        ]

        for root_fret in root_frets_on_string:
            window_lo = max(0, root_fret - fret_span)
            window_hi = root_fret + fret_span
            upper_strings = list(range(root_string - 1, 0, -1))

            # An open string below the root string that sounds the root
            # itself can ring instead of being muted -- it costs no
            # finger and puts the root in the bass an octave down. This
            # only ever fires for a root on string 5 whose pitch class
            # is the open low E, i.e. E-rooted voicings up the neck,
            # which were being drawn with the low E muted for no reason.
            open_root_bass_strings = [] if rootless else [
                s for s in range(root_string + 1, 7)
                if GUITAR_OPEN_STRING_PC[s] == root_pc
            ]

            # See JAZZ_LOW_THIRD_MIN_ROOT_FRET: low root on string 6
            # means no third on string 5.
            ban_low_third = (
                root_string == 6
                and root_fret < low_third_min_root_fret
                and third_pcs
            )
            # Rootless shapes put a guide tone in the bass, so the same
            # muddiness test applies to the anchor itself: a third low
            # on the E string is no better for having no root under it.
            if (rootless and root_string == 6
                    and root_fret < low_third_min_root_fret
                    and (GUITAR_OPEN_STRING_PC[6] + root_fret) % 12 in third_pcs):
                continue

            per_string_options = {}
            for s in upper_strings:
                # Exclude extensions from bass-register strings
                if s >= bass_cutoff_string:
                    allowed_pcs = all_target_pcs - extension_pcs
                else:
                    allowed_pcs = all_target_pcs
                if ban_low_third and s == 5:
                    allowed_pcs = allowed_pcs - third_pcs
                options = []
                for f in range(window_lo, window_hi + 1):
                    pc = (GUITAR_OPEN_STRING_PC[s] + f) % 12
                    if pc in allowed_pcs:
                        options.append((f, pc))
                per_string_options[s] = options

            # The anchor string already sounds one tone, so it counts
            # towards what the upper strings still have to cover. With a
            # root anchor that tone was the root and was excluded from
            # the requirement anyway; a guide-tone anchor is a required
            # tone, so seeding it is what makes rootless shapes findable
            # at all.
            anchor_pitch_class = (
                GUITAR_OPEN_STRING_PC[root_string] + root_fret) % 12
            upper_required = required_pcs - {anchor_pitch_class}

            def _enumerate(strings_left, chosen, pcs_covered):
                if not strings_left:
                    if not upper_required.issubset(pcs_covered):
                        return
                    frets = {root_string: root_fret}
                    # The anchor is the root in the normal case, but a
                    # guide tone when rootless, so its label comes from
                    # the same table as every other note.
                    anchor_pc = (GUITAR_OPEN_STRING_PC[root_string]
                                 + root_fret) % 12
                    labels = {root_string: pc_to_label.get(anchor_pc, '1')}
                    for s, f in chosen:
                        frets[s] = f
                        pc = (GUITAR_OPEN_STRING_PC[s] + f) % 12
                        labels[s] = pc_to_label.get(pc, str((pc - root_pc) % 12))
                    played_upper = {s for s, _ in chosen}
                    for s in upper_strings:
                        if s not in played_upper:
                            frets[s] = 'x'
                    for s in range(root_string + 1, 7):
                        frets[s] = 'x'
                    played_frets = [f for f in frets.values() if f != 'x']
                    min_fret = min(played_frets)
                    max_fret = max(played_frets)
                    span = max_fret - min_fret
                    if span > fret_span:
                        return
                    # Reject voicings with repeated pitch classes -- a
                    # note appearing on two strings wastes a finger and
                    # adds no new harmonic information.
                    played_pcs = [
                        (GUITAR_OPEN_STRING_PC[s] + f) % 12
                        for s, f in frets.items() if f != 'x'
                    ]
                    if len(played_pcs) != len(set(played_pcs)):
                        return
                    tones = {anchor_pc} | {
                        (GUITAR_OPEN_STRING_PC[s] + f) % 12
                        for s, f in chosen
                    }
                    results.append({
                        'frets': frets,
                        'lowest_fret': min_fret,
                        'labels': labels,
                        'root_string': root_string,
                        'root_fret': root_fret,
                        'tones_present': tones,
                        'min_fret': min_fret,
                        'max_fret': max_fret,
                        'span': span,
                    })

                    # Same shape again with the open root ringing in the
                    # bass, kept as a separate voicing so the muted
                    # version is still offered. Built after the checks
                    # above deliberately: the open string doubles the
                    # root pitch class, which the no-repeats rule would
                    # otherwise reject, and it must stay out of the
                    # span and lowest_fret arithmetic or a shape at
                    # fret 7 would look like it spans seven frets and
                    # render as an open-position diagram.
                    for s_open in open_root_bass_strings:
                        open_frets = dict(frets)
                        open_frets[s_open] = 0
                        open_labels = dict(labels)
                        open_labels[s_open] = '1'
                        results.append({
                            'frets': open_frets,
                            'lowest_fret': min_fret,
                            'labels': open_labels,
                            'root_string': root_string,
                            'root_fret': root_fret,
                            'tones_present': tones,
                            'min_fret': min_fret,
                            'max_fret': max_fret,
                            'span': span,
                        })
                    return

                s = strings_left[0]
                rest = strings_left[1:]
                _enumerate(rest, chosen, pcs_covered)
                for f, pc in per_string_options[s]:
                    _enumerate(rest, chosen + [(s, f)], pcs_covered | {pc})

            _enumerate(upper_strings, [], {anchor_pitch_class})

    # ------------------------------------------------------------------
    # Open-root pedal voicings.
    #
    # The pass above ties the whole shape to a window around the root's
    # own fret, so an open root only ever gets a grip within four frets
    # of the nut. But an open string costs no finger and can ring under
    # a shape played anywhere on the neck: A7b9 as x-0-5-6-5-6, with the
    # open A under a grip at the fifth fret, is unreachable from a
    # window centred on fret 0. (E-rooted chords got a taste of this
    # from open_root_bass_strings above, but only as a doubled root
    # beneath a shape already anchored on string 5.)
    #
    # So for a root that matches an open string, sweep the fretted
    # window up the neck independently and let the open string carry the
    # root. Everything below the open string is muted; the open root is
    # kept out of the span arithmetic, exactly as above.
    # ------------------------------------------------------------------
    for open_string in (root_strings if include_open_pedal else ()):
        if GUITAR_OPEN_STRING_PC[open_string] != root_pc:
            continue

        upper_strings = list(range(open_string - 1, 0, -1))
        upper_required = required_pcs - {root_pc}

        for window_lo in range(1, 13):
            window_hi = window_lo + fret_span

            per_string_options = {}
            for s in upper_strings:
                if s >= bass_cutoff_string:
                    allowed_pcs = all_target_pcs - extension_pcs
                else:
                    allowed_pcs = all_target_pcs
                # The open root is at the nut, so the muddy-low-third
                # rule applies here for a string-6 root as it would at
                # any root fret below the threshold.
                if open_string == 6 and s == 5 and third_pcs \
                        and low_third_min_root_fret > 0:
                    allowed_pcs = allowed_pcs - third_pcs
                options = []
                for f in range(window_lo, window_hi + 1):
                    pc = (GUITAR_OPEN_STRING_PC[s] + f) % 12
                    if pc in allowed_pcs:
                        options.append((f, pc))
                per_string_options[s] = options

            def _enumerate_pedal(strings_left, chosen, pcs_covered):
                if not strings_left:
                    if not upper_required.issubset(pcs_covered):
                        return
                    if len(chosen) < 2:
                        return          # an open root plus one note is not a voicing
                    fretted = [f for _, f in chosen]
                    min_fret = min(fretted)
                    max_fret = max(fretted)
                    if max_fret - min_fret > fret_span:
                        return
                    # Only keep shapes that actually sit up the neck --
                    # anything reachable from the nut is already covered
                    # by the main pass.
                    if min_fret <= fret_span:
                        return
                    played_pcs = [root_pc] + [
                        (GUITAR_OPEN_STRING_PC[s] + f) % 12 for s, f in chosen
                    ]
                    if len(played_pcs) != len(set(played_pcs)):
                        return

                    frets = {open_string: 0}
                    labels = {open_string: '1'}
                    for s, f in chosen:
                        frets[s] = f
                        pc = (GUITAR_OPEN_STRING_PC[s] + f) % 12
                        labels[s] = pc_to_label.get(pc, str((pc - root_pc) % 12))
                    played_upper = {s for s, _ in chosen}
                    for s in upper_strings:
                        if s not in played_upper:
                            frets[s] = 'x'
                    for s in range(open_string + 1, 7):
                        frets[s] = 'x'

                    results.append({
                        'frets': frets,
                        'lowest_fret': min_fret,
                        'labels': labels,
                        'root_string': open_string,
                        'root_fret': 0,
                        'tones_present': set(played_pcs),
                        'min_fret': min_fret,
                        'max_fret': max_fret,
                        'span': max_fret - min_fret,
                        'open_root_pedal': True,
                    })
                    return

                s = strings_left[0]
                rest = strings_left[1:]
                _enumerate_pedal(rest, chosen, pcs_covered)
                for f, pc in per_string_options[s]:
                    _enumerate_pedal(rest, chosen + [(s, f)], pcs_covered | {pc})

            _enumerate_pedal(upper_strings, [], set())

    # Deduplicate, treating octave twins as the same voicing. Two shapes
    # on the same strings whose frets all agree mod 12 are the identical
    # chord twelve frets apart -- the open A7b9 x-0-2-3-2-3 and its
    # x-12-14-15-14-15 copy are one idea, not two, and printing both
    # wastes half the sheet on E- and A-rooted chords. Within a span of
    # four frets, agreeing mod 12 can only mean an exact transposition
    # by an octave, so this cannot merge genuinely different grips.
    # The survivor is whichever sits nearest the nut, counting an open
    # string as fret 0, so an open or pedal version always wins over its
    # copy up the neck.
    def _octave_key(v):
        return tuple(sorted(
            (s, f % 12) for s, f in v['frets'].items() if f != 'x'
        ))

    def _nut_distance(v):
        played = [f for f in v['frets'].values() if f != 'x']
        return (min(played), sum(1 for f in v['frets'].values() if f == 'x'))

    best_by_key = {}
    order = []
    for v in results:
        key = _octave_key(v)
        if key not in best_by_key:
            best_by_key[key] = v
            order.append(key)
        elif _nut_distance(v) < _nut_distance(best_by_key[key]):
            best_by_key[key] = v
    unique = [best_by_key[k] for k in order]

    unique.sort(key=lambda v: (
        sum(1 for f in v['frets'].values() if f == 'x'),
        v['span'],
        v['root_fret'],
    ))

    from collections import defaultdict
    buckets = defaultdict(list)
    for v in unique:
        buckets[(v['root_string'], v.get('open_root_pedal', False))].append(v)

    # Pedal voicings get their own buckets rather than competing with the
    # movable shapes: they mute only the strings below the open root, so
    # they would otherwise sort straight to the top and crowd the
    # fretted shapes out of the list entirely.
    final = []
    for rs in (6, 5):
        final.extend(buckets[(rs, False)][:max_voicings])
    for rs in (6, 5):
        final.extend(buckets[(rs, True)][:max_voicings])
    return final


def draw_jazz_shapes(parsed_root, chord_name, base_slug,
                     optional_fifth=True, fret_span=4, max_voicings=4,
                     include_open_pedal=False, rootless=False):
    """
    Find and draw practical jazz voicings for an arbitrary chord
    spelling, with roots on strings 6 and 5.

    Uses assign_octaves on the user's spelling to determine which tones
    are close-voiced chord tones vs. upper-register extensions:
      - Notes within the first octave of the root (e.g. the A in
        'C E A D', abs interval <= 12st) are chord tones and can
        appear anywhere in the voicing.
      - Notes above the octave (e.g. the D in 'C E A D', abs interval
        14st = a 9th) are extensions, restricted to strings 1-3 only.
    Labels follow the same logic: A is '6' not '13', D is '9' not '2'.
    """
    root_pc = (PITCH_CLASS[parsed_root[0][0]] + parsed_root[0][1]) % 12
    chord_pcs = [(PITCH_CLASS[n[0]] + n[1]) % 12 for n in parsed_root]
    fifth_pc = (root_pc + 7) % 12

    # Determine register intent from the user's own spelling.
    notes_with_octaves = assign_octaves(parsed_root, start_octave=4)
    root_abs = 4 * 12 + root_pc

    root_letter, root_accidental = parsed_root[0][0], parsed_root[0][1]

    # Label from the spelling, not from the pitch class. This used to go
    # through semitone->name tables, which could not tell Ab from G#: an
    # above-octave 8-semitone tone came back '#5' for both, so a b13
    # showed as #5. degree_label works from letter + accidental and is
    # what the keyboard diagrams and harmonized chart already use, so
    # jazz labels now agree with the rest of the app: Ab over C is b13
    # up high and b6 down low, while G# over C stays #5 either way.
    extension_pcs = set()
    pc_to_label = {}
    for n in notes_with_octaves:
        pc = n['pc']
        abs_interval = (n['octave'] * 12 + pc) - root_abs
        above_octave = abs_interval > 12
        if above_octave:
            extension_pcs.add(pc)
        pc_to_label[pc] = degree_label(
            root_letter, root_accidental,
            n['letter'], n['accidental'],
            above_octave=above_octave,
        )
    pc_to_label.setdefault(root_pc, '1')

    optional_pcs = {fifth_pc} if optional_fifth else set()

    voicings = find_jazz_voicings(
        chord_pcs, root_pc,
        root_strings=(6, 5),
        fret_span=fret_span,
        optional_pcs=optional_pcs,
        max_voicings=max_voicings,
        extension_pcs=extension_pcs,
        bass_cutoff_string=4,
        pc_to_label=pc_to_label,
        include_open_pedal=include_open_pedal,
        rootless=rootless,
    )

    if not voicings:
        return []

    images = []
    string_name = {6: 'E', 5: 'A'}
    for i, v in enumerate(voicings):
        rs = v['root_string']
        rf = v['root_fret']
        if v.get('open_root_pedal'):
            root_str_label = f"open {string_name[rs]}-string root"
        else:
            root_str_label = f"str.{rs} ({string_name[rs]}-string)"
        shape_label = f"str.{rs}, fr.{rf}"
        img_base = f"{base_slug}-jazz-{i + 1}"
        draw_fretboard_diagram(
            v['frets'], v['lowest_fret'], v['labels'],
            shape_label, chord_name, img_base
        )
        images.append((f"{chord_name} \u2013 Jazz shape {i + 1} ({root_str_label})", img_base))
    return images


FRETBOARD_FULL_N_FRETS = 12


def draw_full_fretboard(target_pcs_to_label, title, base_filename, root_pc=None):
    """
    Every occurrence of every target pitch class, across all 6 strings
    and frets 0-12 (one full octave plus the open position) -- the
    fretboard equivalent of the keyboard's "full chord" view, and also
    used for a full-neck scale overview. Unlike a CAGED shape, this
    isn't one specific playable fingering; it's a reference map of
    where every relevant tone lives on the neck.

    target_pcs_to_label: dict {pitch_class (0-11): display_label}, e.g.
    {0: 'C', 4: 'E', 7: 'G'} for a C major chord, or all 7 (or however
    many) scale tones for a scale overview.

    root_pc: pitch class (0-11) of the chord root or scale tonic, if
    known. Every occurrence of this pitch class gets its own color so
    the root stands out from the other chord/scale tones at a glance.
    None (the default) draws every tone the same way, as before.
    """
    string_spacing = 1.0
    fret_spacing = 0.85  # slightly tighter than the CAGED diagrams so
                          # 12 frets stay a manageable total width
    margin_top = 0.5
    margin_bottom = 0.5
    margin_left = 0.75
    margin_right = 0.5

    board_w = FRETBOARD_FULL_N_FRETS * fret_spacing
    board_h = (FRETBOARD_N_STRINGS - 1) * string_spacing

    fig_w = (board_w + margin_left + margin_right) * 1.05
    fig_h = (board_h + margin_top + margin_bottom) * 1.05
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    # Horizontal neck orientation here (unlike the vertical CAGED
    # diagrams): x = fret position (0 = nut, increasing right), y =
    # string position (string 6 at top, string 1 at bottom) -- this
    # layout reads more naturally once the whole 12-fret neck is shown
    # at once, matching how players usually picture the full neck.
    def fret_x(fret_num):
        return margin_left + fret_num * fret_spacing

    def string_y(string_num):
        return margin_top + (string_num - 1) * string_spacing

    ax.add_patch(Rectangle((fret_x(0) - 0.04, margin_top - 0.15), 0.08,
                            board_h + 0.3, facecolor=FRETBOARD_NUT_COLOR, edgecolor='none'))
    for f in range(1, FRETBOARD_FULL_N_FRETS + 1):
        x = fret_x(f)
        ax.plot([x, x], [margin_top - 0.15, margin_top + board_h + 0.15],
                color=FRETBOARD_FRET_LINE, linewidth=1.0)

    for s in range(1, FRETBOARD_N_STRINGS + 1):
        y = string_y(s)
        ax.plot([fret_x(0), fret_x(FRETBOARD_FULL_N_FRETS)], [y, y],
                color=FRETBOARD_STRING_COLOR, linewidth=1.2)

    # Standard inlay-fret position markers
    for f in [3, 5, 7, 9, 12]:
        x = (fret_x(f) + fret_x(f - 1)) / 2
        ax.text(x, margin_top + board_h + 0.35, str(f), ha='center', va='top',
                fontsize=8, color=FRETBOARD_LABEL_TEXT)

    dot_radius = 0.27
    for s in range(1, FRETBOARD_N_STRINGS + 1):
        y = string_y(s)
        for f in range(0, FRETBOARD_FULL_N_FRETS + 1):
            pc = (GUITAR_OPEN_STRING_PC[s] + f) % 12
            if pc not in target_pcs_to_label:
                continue
            # Open-string dots sit clearly LEFT of the nut rather than
            # centered on it -- centering them on the nut line puts
            # them close enough to the fret-1 cell's dot that the two
            # visually overlap (their separation is less than one dot
            # diameter at this scale).
            x_center = (fret_x(0) - 0.32) if f == 0 else (fret_x(f) + fret_x(f - 1)) / 2
            dot = Circle((x_center, y), dot_radius, facecolor=FRETBOARD_HL_FILL,
                         edgecolor=FRETBOARD_HL_STROKE, linewidth=1.1, zorder=3)
            ax.add_patch(dot)
            label = target_pcs_to_label[pc]
            ax.text(x_center, y, label, ha='center', va='center', fontsize=7.5,
                    color=FRETBOARD_HL_TEXT, fontweight='bold', zorder=4)

    ax.set_xlim(margin_left - 0.55, board_w + margin_left + margin_right)
    ax.set_ylim(margin_top + board_h + 0.6, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text((fret_x(0) + fret_x(FRETBOARD_FULL_N_FRETS)) / 2, -0.35, title,
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color=FRETBOARD_LABEL_TEXT)

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight', transparent=True)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)


def draw_full_fretboard_chord(parsed_root, chord_name, base_slug):
    """
    Full-fretboard view for a chord -- every occurrence of every chord
    tone, labeled by letter name (matching the keyboard's full-chord
    style), with the chord's root tone highlighted in its own color.
    Unlike CAGED, this works for ANY chord quality (not just plain
    major/minor triads), since it's just a pitch-class lookup, no
    shape templates involved.
    """
    target_pcs = {}
    for letter, accidental, pc, display_name in parsed_root:
        target_pcs[pc] = display_name
    root_pc = parsed_root[0][2]
    img_base = f"{base_slug}-fretboard-full"
    draw_full_fretboard(target_pcs, f"{chord_name} \u2013 Full fretboard", img_base, root_pc=root_pc)
    return [(f"{chord_name} \u2013 Full fretboard (guitar)", img_base)]


def draw_full_fretboard_scale(parsed_scale, scale_name, base_slug):
    """
    Full-fretboard view for a scale -- every occurrence of every scale
    tone across the neck, labeled by letter name, with the tonic
    highlighted in its own color.
    """
    target_pcs = {}
    for letter, accidental, pc, display_name in parsed_scale:
        target_pcs[pc] = display_name
    root_pc = parsed_scale[0][2]
    img_base = f"{base_slug}-fretboard-full"
    draw_full_fretboard(target_pcs, f"{scale_name} \u2013 Full fretboard", img_base, root_pc=root_pc)
    return [(f"{scale_name} \u2013 Full fretboard (guitar)", img_base)]


# ----------------------------------------------------------------------
# Reusable chord rendering (used by both manually-typed chords and
# auto-generated chords from a harmonized scale)
# ----------------------------------------------------------------------

def render_chord_with_inversions(parsed_root, name, slug, ask_inversions=True,
                                  how_many=None, want_full_chord=None,
                                  want_info_panel=None, want_guitar_shapes=None,
                                  want_jazz_shapes=None, want_full_fretboard=None,
                                  want_pedal_voicings=False,
                                  rootless_guitar=False, rootless_keyboard=False):
    """
    Runs the full chord-rendering flow used for a single chord: prompts
    for how many inversions (unless ask_inversions=False, e.g. for a
    chord with only one possible voicing), generates each inversion's
    keyboard image + LilyPond block with consistent sizing, then offers
    the 5-octave "full chord" view and a diatonic-function info panel.
    Returns (keyboard_images, ly_blocks) in the same shape
    process_one_entry already accumulates.

    This is the same logic chord mode has always used for a manually
    typed chord -- pulled out into its own function so a harmonized
    scale's chords (built automatically from the scale, not typed by
    hand) can go through the identical rendering path: same inversion
    prompts, same degree labels, same keyboard image style.

    how_many / want_full_chord / want_info_panel: pass these to skip
    the interactive input() prompts entirely (e.g. when called from a
    GUI, which collects every answer up front via a form instead of a
    live terminal session). When left as None, behavior is unchanged
    from before -- prompts interactively.
    """
    keyboard_images = []
    ly_blocks = []

    max_inv = len(parsed_root)
    if how_many is not None:
        how_many = max(1, min(int(how_many), max_inv))
    elif ask_inversions and max_inv > 1:
        how_many_raw = input(
            f"How many inversions for {name}? [1-{max_inv}, default {max_inv}]: "
        ).strip()
        try:
            how_many = int(how_many_raw) if how_many_raw else max_inv
        except ValueError:
            how_many = max_inv
        how_many = max(1, min(how_many, max_inv))
    else:
        how_many = max_inv

    root_letter, root_accidental = parsed_root[0][0], parsed_root[0][1]

    # Rootless keyboard voicings: drop the root before the inversions
    # are generated, so each inversion is a rotation of what actually
    # sounds. The root letter above is kept as the reference the degree
    # labels are measured from, so they still read 3 / b7 / 9 / 13.
    voiced_notes = parsed_root
    if rootless_keyboard:
        reference_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
        voiced_notes = [
            note for note in parsed_root[1:]
            if (PITCH_CLASS[note[0]] + note[1]) % 12 != reference_pc
        ]
        if not voiced_notes:
            voiced_notes = parsed_root
        how_many = max(1, min(how_many, len(voiced_notes)))

    inversions = generate_inversions(voiced_notes, how_many)

    # Compute which pitch classes are extensions once, from the root-
    # position voicing -- the user's spelling determines harmonic
    # register intent, and that intent should be consistent across all
    # inversions. Without this, an inversion that places Bbb near the
    # bottom of the voicing would wrongly label it 'b2' instead of
    # 'b9', since it's no longer > 1 octave above the root in that
    # inversion's absolute pitch layout.
    root_position_voicing = inversions[0]
    root_pc = (PITCH_CLASS[root_letter] + root_accidental) % 12
    root_ref_abs = 4 * 12 + root_pc
    extension_pcs = {
        n["pc"] for n in root_position_voicing
        if (n["octave"] * 12 + n["pc"]) - root_ref_abs > 12
    }

    # As in draw_keyboard_inversions: each inversion gets the narrowest
    # window that fits it, rather than every inversion inheriting the
    # width of the widest one.

    for k, notes_with_octaves in enumerate(inversions):
        inv_label = inversion_name(k)
        degree_labels = degree_labels_for_voicing(
            notes_with_octaves, root_letter, root_accidental,
            extension_pcs=extension_pcs
        )

        print(f"{name} -- {inv_label}:")
        for n, d in zip(notes_with_octaves, degree_labels):
            print(f"  {n['display_name']}{n['octave']}  ({d})")

        img_base = f"{slug}-{slugify(inv_label)}"
        draw_keyboard(notes_with_octaves, degree_labels, img_base)
        keyboard_images.append((f"{name} \u2013 {inv_label}", img_base))
        ly_blocks.append(ly_chord_block(inv_label, notes_with_octaves,
                                         above=(k % 2 == 0)))

    if want_full_chord is not None:
        full_raw = "y" if want_full_chord else "n"
    else:
        try:
            full_raw = input(f"Full chord view across 5 octaves for {name}? [y/N]: ").strip().lower()
        except EOFError:
            full_raw = ""
    if full_raw in ("y", "yes"):
        full_voicing = generate_full_chord_voicing(voiced_notes, 5, start_octave=3)
        full_degree_labels = degree_labels_for_voicing(
            full_voicing, root_letter, root_accidental,
            extension_pcs=extension_pcs
        )
        full_img_base = f"{slug}-full-chord"
        draw_keyboard(full_voicing, full_degree_labels, full_img_base, min_octaves=5)
        keyboard_images.append((f"{name} \u2013 Full chord", full_img_base))

        ly_full = generate_full_chord_voicing(voiced_notes, 2, start_octave=4)
        ly_blocks.append(ly_chord_block("Full chord", ly_full,
                                         above=(len(inversions) % 2 == 0)))

    if want_info_panel is not None:
        info_raw = "y" if want_info_panel else "n"
    else:
        try:
            info_raw = input(
                f"Show diatonic key/function info for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            info_raw = ""
    if info_raw in ("y", "yes") and len(parsed_root) >= 3:
        # Function lookup is based on the chord's underlying TRIAD
        # (root, 3rd, 5th) regardless of how many notes the actual
        # chord has -- a 7th chord's diatonic "role" is still defined
        # by its triad; the 7th doesn't change which key degree it is.
        triad_pcs = [
            (PITCH_CLASS[n[0]] + n[1]) % 12 for n in parsed_root[:3]
        ]
        _, triad_quality_abbrev = classify_triad(*triad_pcs)
        functions = find_chord_functions(root_letter, root_accidental, triad_quality_abbrev)
        triad_quality_name, _ = classify_triad(*triad_pcs)
        info_img_base = f"{slug}-info"
        draw_chord_info_panel(name, triad_quality_name, functions, info_img_base)
        keyboard_images.append((f"{name} \u2013 Diatonic function", info_img_base))

    if want_guitar_shapes is not None:
        guitar_raw = "y" if want_guitar_shapes else "n"
    else:
        try:
            guitar_raw = input(
                f"Show guitar fretboard (CAGED shapes) for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            guitar_raw = ""
    if guitar_raw in ("y", "yes") and len(parsed_root) >= 3:
        # CAGED is a triad-shape system, so it's the chord's underlying
        # triad quality (root/3rd/5th) that determines which 5 shapes
        # apply -- same reasoning as the info panel above. Reuses
        # triad_pcs/triad_quality_abbrev if the info panel already
        # computed them this call, otherwise computes them fresh.
        triad_pcs_g = [(PITCH_CLASS[n[0]] + n[1]) % 12 for n in parsed_root[:3]]
        _, triad_quality_abbrev_g = classify_triad(*triad_pcs_g)
        if triad_quality_abbrev_g not in ('maj', 'min'):
            print(f"  (No CAGED shapes available for a {triad_quality_abbrev_g} "
                  f"triad -- CAGED only covers major and minor chords.)")
        else:
            guitar_images = draw_all_caged_shapes(
                root_letter, root_accidental, triad_quality_abbrev_g, name, slug
            )
            keyboard_images.extend(guitar_images)

    if want_jazz_shapes is not None:
        jazz_raw = "y" if want_jazz_shapes else "n"
    else:
        try:
            jazz_raw = input(
                f"Show jazz guitar shapes (roots on strings 6 & 5) for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            jazz_raw = ""
    if jazz_raw in ("y", "yes"):
        jazz_images = draw_jazz_shapes(
            parsed_root, name, slug,
            include_open_pedal=want_pedal_voicings,
            rootless=rootless_guitar,
        )
        if jazz_images:
            keyboard_images.extend(jazz_images)
        else:
            print(f"  (No jazz voicings found for {name} within a 4-fret span.)")

    if want_full_fretboard is not None:
        full_fret_raw = "y" if want_full_fretboard else "n"
    else:
        try:
            full_fret_raw = input(
                f"Show full fretboard (every occurrence, all 12 frets) for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            full_fret_raw = ""
    if full_fret_raw in ("y", "yes"):
        # Works for ANY chord quality, unlike CAGED -- this is a plain
        # pitch-class lookup across the neck, not a named shape, so
        # there's no restriction to major/minor triads here.
        fretboard_images = draw_full_fretboard_chord(parsed_root, name, slug)
        keyboard_images.extend(fretboard_images)

    return keyboard_images, ly_blocks


# ----------------------------------------------------------------------
# Process one entry
# ----------------------------------------------------------------------

def process_one_entry(entry_num):
    """
    Returns a list of sub-entries: [(sub_name, ly_blocks, keyboard_images), ...]
    Each sub-entry gets its OWN staff and its own page in the final
    document -- this matters once a scale's harmonized chords are also
    being rendered, since 7+ chords' worth of inversions packed onto one
    shared staff becomes an unreadable wall of notation. A plain chord
    entry (mode 'c') always returns exactly one sub-entry; a scale
    entry returns one sub-entry for the scale itself, plus one more per
    harmonized chord the person chooses to render.
    """
    print(f"\n--- Chord/scale #{entry_num} ---")
    mode = input("Chord or scale? [c/s]: ").strip().lower()
    while mode not in ("c", "s", "chord", "scale"):
        mode = input("Please type 'c' or 's': ").strip().lower()
    mode = "c" if mode in ("c", "chord") else "s"

    label = "Chord" if mode == "c" else "Scale"
    name = input(f"{label} name (e.g. 'Ab major'): ").strip() or f"Untitled {label.lower()}"
    slug = slugify(name)

    sub_entries = []  # list of (sub_name, ly_blocks, keyboard_images)

    if mode == "c":
        parsed_root = read_notes_prompt("Root-position spelling, low to high (e.g. 'Ab C Eb'): ")
        new_images, new_blocks = render_chord_with_inversions(parsed_root, name, slug)
        sub_entries.append((name, new_blocks, new_images, []))

    else:  # Scale
        parsed = read_notes_prompt("Scale spelling, tonic first (e.g. 'C D E F G A B C'): ")
        notes_with_octaves = assign_octaves(parsed, start_octave=4, mode='scale')
        scale_blocks = [ly_scale_block(name, notes_with_octaves)]

        try:
            scale_fretboard_raw = input(
                f"Show full guitar fretboard (every occurrence, all 12 frets) "
                f"for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            scale_fretboard_raw = ""
        if scale_fretboard_raw in ("y", "yes"):
            scale_images.extend(draw_full_fretboard_scale(parsed, name, slug))

        # Full scale across multiple octaves on the keyboard -- the
        # scale equivalent of a chord's "full chord view". A scale
        # spelling normally ends on a repeated tonic (e.g. "C D E F G
        # A B C"); repeating THAT note set verbatim across octaves
        # would double up the tonic at every octave seam, so the
        # repeat is dropped first, the same way it already is before
        # harmonizing.
        full_scale_notes = parsed
        if len(parsed) > 1:
            first_pc_fs = (PITCH_CLASS[parsed[0][0]] + parsed[0][1]) % 12
            last_pc_fs = (PITCH_CLASS[parsed[-1][0]] + parsed[-1][1]) % 12
            if first_pc_fs == last_pc_fs:
                full_scale_notes = parsed[:-1]
        try:
            full_scale_raw = input(
                f"Show full scale across multiple octaves on the keyboard "
                f"for {name}? [y/N]: "
            ).strip().lower()
        except EOFError:
            full_scale_raw = ""
        if full_scale_raw in ("y", "yes"):
            full_scale_voicing = generate_full_chord_voicing(
                full_scale_notes, FULL_SCALE_KEYBOARD_OCTAVES, start_octave=3
            )
            full_scale_labels = degree_labels_for_voicing(
                full_scale_voicing, root_letter, root_accidental,
                extension_pcs=set(),
            )
            full_scale_img_base = f"{slug}-full-scale"
            draw_keyboard(full_scale_voicing, full_scale_labels, full_scale_img_base,
                          min_octaves=FULL_SCALE_KEYBOARD_OCTAVES)
            scale_images.append((f"{name} \u2013 Full scale", full_scale_img_base))

        # Harmonized chord chart: drop a trailing repeated tonic if present
        # (e.g. "C D E F G A B C" -> harmonize using just the first 7), so
        # the chart builds one triad per distinct scale degree, not one
        # extra row for the octave repeat.
        harmonize_notes = parsed
        if len(parsed) > 1:
            first_pc = (PITCH_CLASS[parsed[0][0]] + parsed[0][1]) % 12
            last_pc = (PITCH_CLASS[parsed[-1][0]] + parsed[-1][1]) % 12
            if first_pc == last_pc:
                harmonize_notes = parsed[:-1]

        # Key signature staff: a small separate staff showing just the
        # clef and key signature (no notes), using whichever of
        # LilyPond's seven built-in church modes matches the scale.
        # Only meaningful for a 7-note scale -- a 5- or 6-note scale
        # doesn't have one conventional key signature to fall back on.
        # This needs its own \score (different engraver setup -- no
        # time signature engraver), so it's tracked separately from
        # scale_blocks, which only holds music for the main staff.
        extra_score_blocks = []
        ks_score_block, ks_message = key_signature_block(harmonize_notes)
        if ks_score_block is not None:
            extra_score_blocks.append(ks_score_block)
        if ks_message:
            print(f"\nNote: {ks_message}")

        if len(harmonize_notes) >= 3:
            try:
                chart_raw = input(
                    "Also generate a harmonized chord chart for this scale? [y/N]: "
                ).strip().lower()
            except EOFError:
                chart_raw = ""
            if chart_raw in ("y", "yes"):
                # A 3-note scale can't support a meaningful 4-note chord
                # (it would have to wrap around and reuse the root as
                # its own 7th), so only offer the triad/seventh choice
                # when there's room for it.
                chord_size = 3
                if len(harmonize_notes) >= 4:
                    try:
                        size_raw = input(
                            "Triads or 4-note (seventh) chords? [t/4, default t]: "
                        ).strip().lower()
                    except EOFError:
                        size_raw = ""
                    chord_size = 4 if size_raw in ("4", "seventh", "7", "four") else 3

                chart_suffix = "harmonized-7ths" if chord_size == 4 else "harmonized"
                chart_label_suffix = "Harmonized 7th chords" if chord_size == 4 else "Harmonized chords"
                chart_base = f"{slug}-{chart_suffix}"
                harmonized_rows = harmonize(harmonize_notes, chord_size=chord_size)
                draw_harmonized_chart(harmonize_notes, name, chart_base,
                                       chord_size=chord_size)
                scale_images.append((f"{name} \u2013 {chart_label_suffix}", chart_base))

                # Offer to render each harmonized chord (and its
                # inversions) on the keyboard, exactly the same way a
                # manually-typed chord would be -- same prompts, same
                # rendering path, just auto-fed from the chart's rows
                # instead of typed by hand. Each chord becomes its OWN
                # sub-entry (own staff, own page) -- it is a genuinely
                # different chord (different root, often different
                # quality) from the parent scale, not a variation of it,
                # so it should never share a staff with the scale or
                # with any other chord.
                try:
                    render_chords_raw = input(
                        "Also render each harmonized chord (with inversions) "
                        "on the keyboard? [y/N]: "
                    ).strip().lower()
                except EOFError:
                    render_chords_raw = ""
                if render_chords_raw in ("y", "yes"):
                    for row in harmonized_rows:
                        # Name the chord by what it actually IS -- its own
                        # root and quality (e.g. "F minor") -- never by
                        # gluing the parent scale's name onto the Roman
                        # numeral. "ii of Eb major" is F minor, not
                        # "Eb major ii"; the numeral says which scale
                        # degree it's built on, it doesn't rename the
                        # chord itself.
                        chord_root_display = row['notes'][0]
                        quality_word = QUALITY_WORD.get(
                            row['quality_abbrev'], row['quality_name'].lower()
                        )
                        chord_name = f"{chord_root_display} {quality_word}"
                        chord_subtitle = f"{row['numeral']} of {name}"
                        full_chord_label = f"{chord_name} ({chord_subtitle})"

                        chord_slug = f"{slug}-{slugify(row['numeral'])}-{slugify(chord_root_display)}"
                        chord_root_notes = [
                            parse_note(note_name) for note_name in row['notes']
                        ]
                        new_images, new_blocks = render_chord_with_inversions(
                            chord_root_notes, full_chord_label, chord_slug
                        )
                        sub_entries.append((full_chord_label, new_blocks, new_images, []))

        sub_entries.insert(0, (name, scale_blocks, scale_images, extra_score_blocks))

    return sub_entries


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("Chord / Scale Diagram Generator")
    print("==============================")

    entries_data = []          # (name, ly_filename, keyboard_images)
    all_keyboard_images = []   # flattened for copying later

    entry_num = 1
    while True:
        sub_entries = process_one_entry(entry_num)

        for sub_name, ly_blocks, keyboard_images, extra_score_blocks in sub_entries:
            sub_slug = slugify(sub_name)
            ly_filename = f"{sub_slug}.ly"
            # Guard against two sub-entries slugifying to the same
            # filename (e.g. two different top-level entries each
            # producing an "F minor" chord) -- append the entry number
            # to keep filenames unique without the person ever seeing
            # a silent overwrite.
            if any(existing_name == ly_filename for _, existing_name, _ in entries_data):
                ly_filename = f"{sub_slug}-{entry_num}.ly"

            ly_source = build_single_lilypond(sub_name, ly_blocks, extra_score_blocks)
            with open(ly_filename, "w") as f:
                f.write(ly_source)
            print(f"Wrote {ly_filename}")

            entries_data.append((sub_name, ly_filename, keyboard_images))
            all_keyboard_images.extend(keyboard_images)

        if input("\nAdd another chord or scale? [y/N]: ").strip().lower() not in ("y", "yes"):
            break
        entry_num += 1

    # Overall document title
    doc_title = entries_data[0][0] if len(entries_data) == 1 else "Chord and Scale Reference"

    default_slug = slugify(doc_title)
    try:
        filename_raw = input(
            f"\nOutput filename (without extension) [{default_slug}]: "
        ).strip()
    except EOFError:
        filename_raw = ""
    output_stem = slugify(filename_raw) if filename_raw else default_slug

    # Guard against a real failure mode: if the chosen output stem is the
    # SAME as one of the keyboard image basenames (e.g. a scale named "A
    # major" -> keyboard image "a-major.pdf", and the default output
    # filename is also "a-major"), pdflatex will try to compile
    # "a-major.tex" into "a-major.pdf" while ALSO needing to read
    # "a-major.pdf" as an \includegraphics input -- the same file is
    # simultaneously the thing being written and the thing being read,
    # which reliably fails with "reading image file failed" once pdflatex
    # starts overwriting it mid-build. Append a suffix to disambiguate.
    image_basenames = {base for _, base in all_keyboard_images}
    if output_stem in image_basenames:
        original_stem = output_stem
        output_stem = f"{output_stem}-doc"
        print(f"(Note: '{original_stem}' collides with a keyboard image "
              f"filename -- using '{output_stem}' for the document instead.)")

    lytex_filename = f"{output_stem}.lytex"
    lytex_source = build_lytex_wrapper(doc_title, entries_data)

    with open(lytex_filename, "w") as f:
        f.write(lytex_source)
    print(f"\nWrote {lytex_filename}")

    print("\nRunning lilypond-book...")
    success, message = try_run_lilypond_book(lytex_filename, all_keyboard_images)
    print(message)


if __name__ == "__main__":
    main()
