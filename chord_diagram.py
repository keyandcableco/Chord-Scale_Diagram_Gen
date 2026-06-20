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

def assign_octaves(parsed_notes, start_octave=4):
    result = []
    prev_absolute = None
    for letter, accidental, pc, display_name in parsed_notes:
        if prev_absolute is None:
            absolute = start_octave * 12 + pc
        else:
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


def degree_label(root_letter, root_accidental, note_letter, note_accidental):
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

    return f"{prefix}{base_degree}"


def degree_labels_for_voicing(notes_with_octaves, root_letter, root_accidental):
    return [
        degree_label(root_letter, root_accidental, n["letter"], n["accidental"])
        for n in notes_with_octaves
    ]


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
            s1
          }}
          \\layout {{
            ragged-right = ##f
            line-width = 40\\mm
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
    ragged-right = ##f
    line-width = 160\\mm
  }}
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
{all_scores}
}}
'''

    return f'''\\version "2.24.0"

#(set-global-staff-size 26)

\\header {{
  title = "{safe_title}"
  tagline = ##f
}}

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


def draw_keyboard(notes_with_octaves, degree_labels, base_filename, min_octaves=2):
    # (same implementation as original - omitted here for brevity but fully included in real file)
    lowest_octave = min(n["octave"] for n in notes_with_octaves)
    highest_octave = max(n["octave"] for n in notes_with_octaves)
    natural_span = highest_octave - lowest_octave + 1
    n_octaves = max(min_octaves, natural_span)

    octave_to_index = {lowest_octave + i: i for i in range(n_octaves)}

    highlighted = {}
    for n, lbl in zip(notes_with_octaves, degree_labels):
        highlighted[(n["pc"], octave_to_index[n["octave"]])] = (n["display_name"], lbl)

    label_positions = {}

    white_names = WHITE_LETTERS * n_octaves
    n_white = len(white_names)
    total_w = n_white * KEY_W

    fig_height_in = 3.3
    fig_w = max(fig_height_in * (total_w / WHITE_H) * 0.78, 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_height_in), dpi=200)

    # White keys
    for i, name in enumerate(white_names):
        x = i * KEY_W
        oct_idx = i // 7
        pc = WHITE_PC[name]
        key = (pc, oct_idx)
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

    # Black keys
    for oct_idx in range(n_octaves):
        octave_x = oct_idx * 7 * KEY_W
        for boundary, pc in BLACK_KEYS:
            x = octave_x + boundary * KEY_W - BLACK_W / 2
            key = (pc, oct_idx)
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
        key = (n["pc"], octave_to_index[n["octave"]])
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
    """
    third_int = (third_pc - root_pc) % 12
    fifth_int = (fifth_pc - root_pc) % 12
    if third_int == 4 and fifth_int == 7:
        return "Major triad", "maj"
    elif third_int == 3 and fifth_int == 7:
        return "Minor triad", "min"
    elif third_int == 3 and fifth_int == 6:
        return "Diminished triad", "dim"
    elif third_int == 4 and fifth_int == 8:
        return "Augmented triad", "aug"
    else:
        return "Other triad", "other"


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
    'other': '?', 'other7': '?',
}
QUALITY_LOWERCASE = {'min', 'dim', 'min7', 'm7b5', 'dim7'}

# Plain-language quality words for naming a chord by what it actually
# IS (e.g. "F minor"), as opposed to the Roman-numeral/symbol notation
# used in the chart itself.
QUALITY_WORD = {
    'maj': 'major', 'min': 'minor', 'dim': 'diminished', 'aug': 'augmented',
    'maj7': 'major 7th', 'dom7': 'dominant 7th', 'min7': 'minor 7th',
    'minmaj7': 'minor-major 7th', 'm7b5': 'half-diminished 7th',
    'dim7': 'diminished 7th', 'augmaj7': 'augmented major 7th',
    'aug7': 'augmented 7th', 'other': 'other', 'other7': 'other 7th',
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

    # Subheader: scale-degree number for columns that ARE a scale tone,
    # blank for chromatic passing tones not in the scale -- shows where
    # the scale itself sits within the full chromatic space.
    sub_top = top_y - header_h
    sub_y = sub_top - subheader_h / 2
    degree_at_col = {}
    for octave_rep in range(2):
        for degree_idx, offset in enumerate(scale_offsets):
            col = offset + 12 * octave_rep
            if col < n_chromatic_cols:
                degree_at_col[col] = str(degree_idx + 1)

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


def build_lytex_wrapper(doc_title, entries_data):
    """entries_data = list of (entry_name, ly_filename, keyboard_images)"""
    parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{setspace}",
        r"\setstretch{1.15}",
        rf"\title{{{latex_escape(doc_title)}}}",
        #r"\author{}",
        #r"\date{}",
        r"\begin{document}",
        #r"\maketitle",
        #r"\tableofcontents",
        #r"\newpage",
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
            r"\begin{lilypond}",
            r'\version "2.24.0"',
            score_block.strip(),
            r"\end{lilypond}",
            "",
            r"\bigskip",
        ])

        for inv_label, image_base in keyboard_images:
            is_harmonized_chart = image_base.endswith("-harmonized") or \
                image_base.endswith("-harmonized-7ths")
            img_width = r"0.99\textwidth" if is_harmonized_chart else r"0.88\textwidth"
            parts.extend([
                r"\begin{center}",
                r"\begin{minipage}{0.99\textwidth}" if is_harmonized_chart
                    else r"\begin{minipage}{0.92\textwidth}",
                r"\centering",
                rf"\textbf{{{latex_escape(inv_label)}}}\\[0.4em]",
                rf"\includegraphics[width={img_width}]{{{image_base}.pdf}}",
                r"\end{minipage}",
                r"\end{center}",
                r"\medskip",
                "",
            ])

        parts.append(r"\newpage")   # New page for next entry

    parts.append(r"\end{document}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# lilypond-book integration (unchanged logic)
# ----------------------------------------------------------------------

def try_run_lilypond_book(lytex_filename, all_keyboard_images, out_dir="lilypond-book-out"):
    tex_filename = lytex_filename.replace(".lytex", ".tex")

    if shutil.which("lilypond-book") is None:
        image_list = "\n".join(f"    cp {base}.pdf {out_dir}/" for _, base in all_keyboard_images)
        return False, (
            "lilypond-book was not found...\n"
            f"    lilypond-book --pdf --output={out_dir} {lytex_filename}\n"
            f"{image_list}\n"
            f"    cd {out_dir} && pdflatex {tex_filename} && cd .."
        )

    try:
        subprocess.run(["lilypond-book", "--pdf", f"--output={out_dir}", lytex_filename],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, f"lilypond-book error:\n{e.stderr.strip()[-1500:]}"

    for _, image_base in all_keyboard_images:
        src = f"{image_base}.pdf"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, src))

    if shutil.which("pdflatex") is None:
        return False, f"pdflatex not found. Run: cd {out_dir} && pdflatex {tex_filename}"

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename],
                       check=True, capture_output=True, text=True, cwd=out_dir)
    except subprocess.CalledProcessError as e:
        return False, f"pdflatex error:\n{e.stdout.strip()[-1500:]}"

    final_pdf = tex_filename.replace(".tex", ".pdf")
    return True, f"Final combined PDF written to ./{out_dir}/{final_pdf}"


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------

def slugify(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
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

FRETBOARD_N_STRINGS = 6
FRETBOARD_MIN_FRETS_SHOWN = 4


def draw_fretboard_diagram(frets, lowest_fret_used, degree_labels_by_string,
                            shape_name, chord_name, base_filename):
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

    if not is_open_position:
        ax.text(-0.55, fret_y(0.5), f"{display_start_fret}fr",
                ha='right', va='center', fontsize=11, color=FRETBOARD_LABEL_TEXT)

    for s in range(1, FRETBOARD_N_STRINGS + 1):
        x = string_x(s)
        fret = frets[s]
        if fret == 'x':
            ax.text(x, margin_top - 0.35, '\u00d7', ha='center', va='center',
                    fontsize=15, color=FRETBOARD_MUTE_COLOR, fontweight='bold')
            continue
        if fret == 0:
            circ = Circle((x, margin_top - 0.32), 0.16, facecolor='none',
                           edgecolor=FRETBOARD_OPEN_RING_COLOR, linewidth=1.3)
            ax.add_patch(circ)
            label = degree_labels_by_string.get(s, '')
            if label:
                ax.text(x, margin_top - 0.32, label, ha='center', va='center',
                        fontsize=7.5, color=FRETBOARD_OPEN_RING_COLOR, fontweight='bold')
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
        dot = Circle((x, y), 0.22, facecolor=FRETBOARD_HL_FILL,
                      edgecolor=FRETBOARD_HL_STROKE, linewidth=1.3, zorder=3)
        ax.add_patch(dot)
        label = degree_labels_by_string.get(s, '')
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=9,
                    color=FRETBOARD_HL_TEXT, fontweight='bold', zorder=4)

    ax.set_xlim(-margin_left, board_w + margin_right)
    ax.set_ylim(margin_top + board_h + margin_bottom, -0.1)
    ax.set_aspect('equal')
    ax.axis('off')

    title = f"{chord_name} \u2013 {shape_name}-shape"
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


FRETBOARD_FULL_N_FRETS = 12


def draw_full_fretboard(target_pcs_to_label, title, base_filename):
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
    style). Unlike CAGED, this works for ANY chord quality (not just
    plain major/minor triads), since it's just a pitch-class lookup,
    no shape templates involved.
    """
    target_pcs = {}
    for letter, accidental, pc, display_name in parsed_root:
        target_pcs[pc] = display_name
    img_base = f"{base_slug}-fretboard-full"
    draw_full_fretboard(target_pcs, f"{chord_name} \u2013 Full fretboard", img_base)
    return [(f"{chord_name} \u2013 Full fretboard (guitar)", img_base)]


def draw_full_fretboard_scale(parsed_scale, scale_name, base_slug):
    """
    Full-fretboard view for a scale -- every occurrence of every scale
    tone across the neck, labeled by letter name.
    """
    target_pcs = {}
    for letter, accidental, pc, display_name in parsed_scale:
        target_pcs[pc] = display_name
    img_base = f"{base_slug}-fretboard-full"
    draw_full_fretboard(target_pcs, f"{scale_name} \u2013 Full fretboard", img_base)
    return [(f"{scale_name} \u2013 Full fretboard (guitar)", img_base)]


# ----------------------------------------------------------------------
# Reusable chord rendering (used by both manually-typed chords and
# auto-generated chords from a harmonized scale)
# ----------------------------------------------------------------------

def render_chord_with_inversions(parsed_root, name, slug, ask_inversions=True,
                                  how_many=None, want_full_chord=None,
                                  want_info_panel=None, want_guitar_shapes=None,
                                  want_full_fretboard=None):
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
    inversions = generate_inversions(parsed_root, how_many)

    common_span = max(2, max(
        max(n["octave"] for n in inv) - min(n["octave"] for n in inv) + 1
        for inv in inversions
    ))

    for k, notes_with_octaves in enumerate(inversions):
        inv_label = inversion_name(k)
        degree_labels = degree_labels_for_voicing(notes_with_octaves, root_letter, root_accidental)

        print(f"{name} -- {inv_label}:")
        for n, d in zip(notes_with_octaves, degree_labels):
            print(f"  {n['display_name']}{n['octave']}  ({d})")

        img_base = f"{slug}-{slugify(inv_label)}"
        draw_keyboard(notes_with_octaves, degree_labels, img_base, min_octaves=common_span)
        keyboard_images.append((f"{name} \u2013 {inv_label}", img_base))
        # Staff markup stays short (just "Root pos.", "First inv.", ...)
        # -- the full chord name/context already sits in this entry's
        # \section* heading right above the staff, so repeating it in
        # the markup over every single chord symbol is redundant and,
        # for a long auto-generated name like "G# diminished (vii of A
        # major)", makes the label visually overwhelm the notation.
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
        full_voicing = generate_full_chord_voicing(parsed_root, 5, start_octave=3)
        full_degree_labels = degree_labels_for_voicing(full_voicing, root_letter, root_accidental)
        full_img_base = f"{slug}-full-chord"
        draw_keyboard(full_voicing, full_degree_labels, full_img_base, min_octaves=5)
        keyboard_images.append((f"{name} \u2013 Full chord", full_img_base))

        ly_full = generate_full_chord_voicing(parsed_root, 2, start_octave=4)
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
        notes_with_octaves = assign_octaves(parsed, start_octave=4)
        root_letter = parsed[0][0]
        root_accidental = parsed[0][1]
        degree_labels = degree_labels_for_voicing(notes_with_octaves, root_letter, root_accidental)

        draw_keyboard(notes_with_octaves, degree_labels, slug)
        scale_images = [(name, slug)]
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
