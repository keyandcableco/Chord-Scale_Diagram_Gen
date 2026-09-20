"""
Lead-sheet input format: parsing and LilyPond chart generation.

The format has three kinds of line, freely interleaved:

    {title: Autumn Leaves}          directives -- metadata, or mid-tune
    {key: Bb}                       changes to time/tempo/section
    {time: 4/4}

    | Cm7 | F7 | Bbmaj7 / / Cm7 |   changes -- bars of chord symbols
    |: Am7b5 | D7 | Gm6 | % :|

    Cm7: C Eb G Bb                  definitions -- spellings for diagrams

Directives borrow ChordPro's `{name: value}` syntax; the bar and repeat
conventions come from printed charts. Definitions are the existing song
sheet chord-list format, unchanged, so an old chord list still parses --
it simply produces no bars and therefore no chart.

Deliberately NOT included: a chord-symbol parser. A symbol in the
changes with no matching definition still engraves as a symbol, it just
gets no diagram. Spelling stays the user's job.

Nothing here draws or engraves; build_leadsheet_score turns a parsed
Song into a LilyPond \\score block for the caller to place in a .lytex.
"""

import re


# Bar-line and beat tokens, longest first so that '|:' is not read as
# '|' followed by ':'.
_BARLINE_TOKENS = ['|:', ':|', '||', '|]', '|']
# '|1' and '|2' open a first/second ending, so they must be matched
# before the plain '|'.
_TOKEN_RE = re.compile(r"\|\d+|\|\]|\|\||\|:|:\||\||%|/|[^\s|]+")
_ENDING_RE = re.compile(r"^\|(\d+)$")

_DIRECTIVE_RE = re.compile(r'^\{\s*([a-zA-Z_]+)\s*:\s*(.*?)\s*\}$')

# A definition line: "Name: spelling" or "Name: spelling : 0 1 2".
_DEFINITION_RE = re.compile(r'^([^:|]+):\s*([^:]+?)\s*(?::\s*([\d\s]+))?$')

# Directives that carry over until changed, vs one-shot markers.
_METADATA_KEYS = {'title', 'composer', 'key', 'time', 'tempo', 'transpose'}
_INLINE_KEYS = {'section', 'time', 'tempo'}

DEFAULT_TIME = (4, 4)

# How many bars per engraved line. Four is the chart convention.
BARS_PER_LINE = 4


class LeadSheetError(ValueError):
    """Raised for input the parser cannot make sense of at all."""


def _beats_per_bar(time_sig):
    """
    Beats in a bar, counted in the unit named by the denominator: 4 for
    4/4, 3 for 3/4, 6 for 6/8. Compound meters are counted in their
    written unit rather than in dotted beats, which keeps '/' meaning
    'one more of whatever the denominator is'.
    """
    return time_sig[0]


def _split_beats(n_chords, beats):
    """
    Divide a bar evenly between its chords. When it will not divide
    evenly the earlier chords take the extra beat, so three chords in
    4/4 give 2+1+1 -- what a player would assume from a written chart.
    """
    base = beats // n_chords
    extra = beats % n_chords
    return [base + (1 if i < extra else 0) for i in range(n_chords)]


def parse_leadsheet(text, default_time=DEFAULT_TIME):
    """
    Parse the lead-sheet format into a Song dict:

        {
          'title', 'composer', 'key', 'tempo',       -- strings or None
          'time': (num, den),
          'bars': [Bar, ...],
          'definitions': {name: {'spelling', 'inversions'}},
          'definition_order': [name, ...],
          'warnings': [str, ...],
        }

    Bar = {
      'chords': [{'name': str, 'beats': int}, ...],
      'time': (num, den),          -- meter in force for this bar
      'repeat_start': bool,
      'repeat_end': bool,
      'end_bar': None | '||' | '|]',
      'mark': None | str,          -- section label printed above the bar
      'tempo': None | str,         -- tempo change at this bar
      'ending': None | int,        -- which volta ending this bar is in
      'is_repeat_of_previous': bool,
    }

    Unparseable lines become warnings rather than errors: a chart with
    one bad line should still render the rest.
    """
    song = {
        'title': None,
        'composer': None,
        'key': None,
        'tempo': None,
        'transpose': None,
        'time': tuple(default_time),
        'bars': [],
        'definitions': {},
        'definition_order': [],
        'warnings': [],
    }

    current_time = tuple(default_time)
    pending_mark = None
    pending_tempo = None
    # The volta ending currently open. This persists across lines: an
    # ending routinely runs to a second line of bars, and scoping it to
    # one line silently dropped the continuation out of the repeat.
    current_ending = None
    # Repeat marks arrive attached to bar lines, so '|:' has to be held
    # until the bar it opens actually exists.
    pending_repeat_start = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        directive = _DIRECTIVE_RE.match(line)
        if directive:
            key = directive.group(1).lower()
            value = directive.group(2)
            if key == 'time':
                parsed = _parse_time_signature(value)
                if parsed:
                    current_time = parsed
                    if not song['bars']:
                        song['time'] = parsed
                else:
                    song['warnings'].append(f"Could not read time signature: {value}")
            elif key == 'section':
                pending_mark = value
                # A new section ends any ending still open.
                current_ending = None
            elif key == 'tempo':
                if not song['bars']:
                    # Before any bars this is the tune's tempo and gets
                    # printed once in the header, so it must not also be
                    # left pending for the first bar that comes along.
                    song['tempo'] = value
                    pending_tempo = None
                else:
                    pending_tempo = value
            elif key in _METADATA_KEYS:
                song[key] = value
            else:
                song['warnings'].append(f"Unknown directive: {key}")
            continue

        if '|' in line:
            bars, pending_repeat_start, current_ending = _parse_changes_line(
                line, current_time, song['warnings'], pending_repeat_start,
                current_ending
            )
            for bar in bars:
                if pending_mark is not None:
                    bar['mark'] = pending_mark
                    pending_mark = None
                if pending_tempo is not None and song['bars']:
                    bar['tempo'] = pending_tempo
                    pending_tempo = None
                song['bars'].append(bar)
            continue

        definition = _DEFINITION_RE.match(line)
        if definition:
            name = definition.group(1).strip()
            spelling = definition.group(2).strip()
            inversions = definition.group(3)
            song['definitions'][name] = {
                'spelling': spelling,
                'inversions': (
                    [int(i) for i in inversions.split()] if inversions else None
                ),
            }
            if name not in song['definition_order']:
                song['definition_order'].append(name)
            continue

        song['warnings'].append(f"Could not read line: {line}")

    _resolve_repeat_bars(song)
    return song


def _parse_time_signature(value):
    m = re.match(r'^(\d+)\s*/\s*(\d+)$', value.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _parse_changes_line(line, time_sig, warnings, pending_repeat_start,
                        current_ending=None):
    """
    Turn one line of bars into Bar dicts. Returns (bars, repeat_start
    still pending, ending still open) -- a '|:' at the end of a line
    opens the first bar of the next line, and an ending carries on to
    the next line until something closes it.
    """
    tokens = _TOKEN_RE.findall(line)
    bars = []

    current = None
    repeat_start = pending_repeat_start
    # Which volta ending the bars being read belong to. An ending runs
    # from its '|N' marker until another ending marker, a ':|', a
    # double bar, a new {section:}, or the end of the changes -- it is
    # NOT closed by the end of a line, since endings often run to two
    # lines of bars.

    def _open_bar():
        return {
            'chords': [],
            'time': time_sig,
            'repeat_start': False,
            'repeat_end': False,
            'end_bar': None,
            'mark': None,
            'tempo': None,
            'ending': None,
            'is_repeat_of_previous': False,
        }

    def _close_bar(bar, repeat_end=False, end_bar=None):
        if bar is None:
            return
        if not bar['chords'] and not bar['is_repeat_of_previous']:
            # An empty stretch between two bar lines is just spacing.
            return
        bar['repeat_end'] = bar['repeat_end'] or repeat_end
        bar['end_bar'] = end_bar or bar['end_bar']
        _assign_beats(bar, warnings)
        bars.append(bar)

    for token in tokens:
        ending_marker = _ENDING_RE.match(token)
        if ending_marker:
            _close_bar(current)
            current_ending = int(ending_marker.group(1))
            current = _open_bar()
            current['ending'] = current_ending
            continue

        if token in _BARLINE_TOKENS or token == ':|':
            if token == '|:':
                _close_bar(current)
                current = _open_bar()
                current['repeat_start'] = True
                repeat_start = False
            elif token == ':|':
                _close_bar(current, repeat_end=True)
                current = None
                current_ending = None
            elif token in ('||', '|]'):
                _close_bar(current, end_bar=token)
                current = None
                current_ending = None
            else:                                   # plain '|'
                _close_bar(current)
                current = _open_bar()
                current['ending'] = current_ending
                if repeat_start:
                    current['repeat_start'] = True
                    repeat_start = False
            continue

        if current is None:
            current = _open_bar()
            current['ending'] = current_ending
            if repeat_start:
                current['repeat_start'] = True
                repeat_start = False

        if token == '%':
            current['is_repeat_of_previous'] = True
        elif token == '/':
            if current['chords']:
                last = current['chords'][-1]
                last['beats'] = (last['beats'] or 1) + 1
            else:
                warnings.append("A bar starts with '/' -- no chord to extend")
        else:
            current['chords'].append({'name': token, 'beats': None})

    _close_bar(current)
    return bars, repeat_start, current_ending


def _assign_beats(bar, warnings):
    """
    Fill in each chord's beat count. Chords with no explicit '/' share
    the bar evenly; if any chord was extended with '/', every chord's
    written length is taken literally and only the total is checked.
    """
    beats = _beats_per_bar(bar['time'])
    chords = bar['chords']
    if not chords:
        return

    explicit = any(c['beats'] is not None for c in chords)
    if not explicit:
        for chord, n in zip(chords, _split_beats(len(chords), beats)):
            chord['beats'] = n
        return

    for chord in chords:
        if chord['beats'] is None:
            chord['beats'] = 1

    total = sum(c['beats'] for c in chords)
    if total != beats:
        names = ' '.join(c['name'] for c in chords)
        warnings.append(
            f"Bar '{names}' has {total} beats, expected {beats} -- "
            "adjusted the last chord to fit"
        )
        if total < beats:
            chords[-1]['beats'] += beats - total
        else:
            # Over-filled: take the excess off the end, dropping whole
            # chords if the last one alone cannot absorb it. Clamping a
            # single chord to 1 beat would leave the bar still overfull
            # and LilyPond would silently bar-check-fail on it.
            excess = total - beats
            while excess > 0 and chords:
                last = chords[-1]
                take = min(excess, last['beats'] - 1)
                last['beats'] -= take
                excess -= take
                if excess > 0:
                    excess -= last['beats']
                    chords.pop()
            if not chords:
                chords.append({'name': '?', 'beats': beats})


def _resolve_repeat_bars(song):
    """
    Expand '%' bars by copying the previous bar's chords.

    The copies are marked 'silent': the bar needs the same rhythm as the
    one it repeats, but the chord symbol is not printed again. Reprinting
    it is what the '%' exists to avoid -- a Real Book chart leaves the
    repeated bar empty under the same harmony.
    """
    previous = None
    for bar in song['bars']:
        if bar['is_repeat_of_previous']:
            if previous is None:
                song['warnings'].append("'%' in the first bar -- nothing to repeat")
                bar['chords'] = []
            else:
                bar['chords'] = [
                    dict(c, silent=True) for c in previous['chords']
                ]
        previous = bar


# ----------------------------------------------------------------------
# LilyPond generation
# ----------------------------------------------------------------------

# Beat counts that map to a single note value, in the denominator's
# unit. Anything else is written as tied notes.
_DURATION_FOR_BEATS = {
    1: ['{d}'],
    2: ['{h}'],
    3: ['{h}.'],
    4: ['{w}'],
    6: ['{w}.'],
    8: ['{w}', '{w}'],
}


def _durations(beats, denominator):
    """
    LilyPond duration strings for a span of `beats` beats where one beat
    is a 1/denominator note. Returns a list -- more than one entry means
    the span needs tied/consecutive rests.
    """
    unit = denominator
    names = {
        '{d}': str(unit),
        '{h}': str(unit // 2) if unit >= 2 else None,
        '{w}': str(unit // 4) if unit >= 4 else None,
    }
    template = _DURATION_FOR_BEATS.get(beats)
    if template:
        out = []
        for piece in template:
            base = piece.rstrip('.')
            dotted = piece.endswith('.')
            value = names.get(base)
            if value is None:
                out = None
                break
            out.append(value + ('.' if dotted else ''))
        if out:
            return out

    # Fall back to repeating single beats, which always works even if it
    # engraves as several rests rather than one long one.
    return [str(unit)] * max(1, beats)


def _escape_markup(text):
    """Quote a chord symbol for LilyPond markup."""
    return text.replace('\\', '').replace('"', '')


def build_leadsheet_score(song, staff_size=18, bars_per_line=BARS_PER_LINE,
                          melody=None):
    """
    Build a LilyPond \\score block for the chart.

    Chord symbols are markup attached to transparent rests rather than a
    ChordNames context: LilyPond's chord namer works from pitches and
    would re-spell the symbols (printing #5 where the user wrote b13),
    and there is no chord-symbol parser here to feed it. Markup prints
    exactly what was typed.

    Transparent rests rather than spacer rests because spacers carry no
    horizontal width -- a chart of nothing but spacers collapses to a
    few millimetres. Proportional spacing with uniform stretching then
    gives every bar the same width, as on a printed chart.

    melody: optional LilyPond note string. When given it replaces the
    invisible rests, and the chord symbols ride above it. Not wired to
    the form yet; the staff is built this way so adding it later is a
    substitution, not a restructuring.
    """
    if not song['bars']:
        return None

    lines = []
    lines.append(r'\score {')
    lines.append(r'  \new Staff \with {')
    if melody is None:
        lines.append(r'    \override Rest.transparent = ##t')
        lines.append(r'    \override MultiMeasureRest.transparent = ##t')
        # Dots are a separate grob from the rest they belong to, so a
        # dotted rest leaves a visible dot floating in the bar unless
        # they are hidden too.
        lines.append(r'    \override Dots.transparent = ##t')
    lines.append(r'    \override TextScript.self-alignment-X = #LEFT')
    lines.append(r'    \override TextScript.X-offset = #-0.6')
    lines.append(r'    \override TextScript.staff-padding = #1.5')
    lines.append(r'    \override TextScript.outside-staff-priority = #450')
    lines.append(r'  } {')

    key_block = _key_block(song.get('key'), song['warnings'])
    if key_block:
        lines.append('    ' + key_block)
    num, den = song['time']
    lines.append(rf'    \time {num}/{den}')
    if song.get('tempo'):
        tempo = _tempo_block(song['tempo'], den)
        if tempo:
            lines.append('    ' + tempo)

    if melody is not None:
        lines.append('    ' + melody)
    else:
        lines.extend(_bar_lines(song, bars_per_line))

    lines.append(r'  }')
    lines.append(r'  \layout {')
    lines.append(r'    ragged-right = ##f')
    # A final line holding one or two bars should not be stretched
    # across the page like a full system.
    lines.append(r'    ragged-last = ##t')
    lines.append(r'    indent = 0\mm')
    lines.append(rf'    #(layout-set-staff-size {staff_size})')
    lines.append(r'    \context {')
    lines.append(r'      \Score')
    lines.append(r'      proportionalNotationDuration = #(ly:make-moment 1/8)')
    lines.append(r'      \override SpacingSpanner.uniform-stretching = ##t')
    lines.append(r'      \override RehearsalMark.outside-staff-priority = #1500')
    lines.append(r'      \override RehearsalMark.self-alignment-X = #LEFT')
    lines.append(r'      \override RehearsalMark.padding = #1.5')
    lines.append(r'    }')
    lines.append(r'  }')
    lines.append(r'}')
    return '\n'.join(lines)


def _repeat_regions(bars):
    """
    Locate volta repeats that use first/second endings.

    Returns {start_index: region} where region is
    {'body_end': i, 'groups': [[indices], ...], 'last': i}. A repeat
    with no ending markers is not listed here -- those still get the
    simple '\\repeat volta 2 { ... }' treatment driven by ':|'.
    """
    regions = {}
    for index, bar in enumerate(bars):
        if not bar['repeat_start']:
            continue

        groups = []
        body_end = None
        current_ending = None
        last = index

        for j in range(index, len(bars)):
            ending = bars[j]['ending']
            if ending is None:
                if groups:
                    break           # past the endings; repeat is over
                continue
            if body_end is None:
                body_end = j - 1
            if ending != current_ending:
                groups.append([])
                current_ending = ending
            groups[-1].append(j)
            last = j

        if groups:
            regions[index] = {
                'body_end': body_end if body_end is not None else index - 1,
                'groups': groups,
                'last': last,
            }
    return regions


def _bar_lines(song, bars_per_line):
    """
    Emit the bars, wrapping repeats and endings.

    Braces are worked out in a prepass rather than bar by bar, because
    an \\alternative block spans several bars and has to close its body,
    open the alternative, and open and close each ending group at the
    right boundaries.
    """
    out = []
    bars = song['bars']
    regions = _repeat_regions(bars)

    # index -> strings emitted before / after that bar
    prefix = {i: [] for i in range(len(bars))}
    suffix = {i: [] for i in range(len(bars))}
    in_ending_region = set()

    for start, region in regions.items():
        n_endings = len(region['groups'])
        prefix[start].insert(0, r'\repeat volta %d {' % max(2, n_endings))
        first_ending = region['groups'][0][0]
        suffix[first_ending - 1].append('}')
        suffix[first_ending - 1].append(r'\alternative {')
        for group in region['groups']:
            prefix[group[0]].append('{')
            suffix[group[-1]].append('}')
        suffix[region['last']].append('}')
        for i in range(start, region['last'] + 1):
            in_ending_region.add(i)

    open_repeat = False
    bars_on_line = 0

    for index, bar in enumerate(bars):
        pieces = list(prefix[index])

        if bar['mark']:
            pieces.append(
                r'\mark \markup { \box \bold "%s" }' % _escape_markup(bar['mark'])
            )
        if bar['tempo']:
            tempo = _tempo_block(bar['tempo'], bar['time'][1])
            if tempo:
                pieces.append(tempo)

        if bar['repeat_start'] and index not in regions:
            if _has_matching_repeat_end(bars, index):
                pieces.append(r'\repeat volta 2 {')
                open_repeat = True
            else:
                pieces.append(r'\bar ".|:"')
                song['warnings'].append(
                    "'|:' with no matching ':|' -- drew the bar line only"
                )

        if bar['ending'] is not None and index not in in_ending_region:
            song['warnings'].append(
                f"Ending {bar['ending']} is not inside a '|:' repeat -- "
                "wrote the bar without a bracket"
            )

        pieces.append(_bar_body(bar))

        if bar['repeat_end'] and index not in in_ending_region:
            if open_repeat:
                pieces.append('}')
                open_repeat = False
            else:
                pieces.append(r'\bar ":|."')
        elif bar['end_bar'] == '||':
            pieces.append(r'\bar "||"')
        elif bar['end_bar'] == '|]':
            pieces.append(r'\bar "|."')
        elif index == len(bars) - 1 and index not in in_ending_region:
            pieces.append(r'\bar "|."')

        # The break goes before the closing braces, not after them:
        # \alternative takes a bare sequence of { } groups, and a
        # \break sitting between two groups makes LilyPond junk the
        # second one ("More alternatives than repeats"). Inside the
        # group it is fine.
        bars_on_line += 1
        if bars_on_line >= bars_per_line and index != len(bars) - 1:
            pieces.append(r'\break')
            bars_on_line = 0

        pieces.extend(suffix[index])

        out.append('    ' + ' '.join(pieces))

    if open_repeat:
        out.append('    }')
        song['warnings'].append("'|:' left open -- closed it at the end")
    return out


def _has_matching_repeat_end(bars, start_index):
    """Whether a '|:' at start_index is ever closed by a ':|'."""
    for offset, bar in enumerate(bars[start_index:]):
        if bar['repeat_end']:
            return True
        if offset and bar['repeat_start']:
            return False
    return False


def _bar_body(bar):
    """One bar of invisible rests carrying the chord symbols."""
    den = bar['time'][1]
    if not bar['chords']:
        return f'R{den * 4 // den if False else _whole_bar_duration(bar)}'

    pieces = []
    for chord in bar['chords']:
        durations = _durations(chord['beats'], den)
        first = True
        for duration in durations:
            if first and not chord.get('silent'):
                pieces.append(
                    'r%s^\\markup { \\bold "%s" }'
                    % (duration, _escape_markup(chord['name']))
                )
            else:
                pieces.append(f'r{duration}')
            first = False
    return ' '.join(pieces)


def _whole_bar_duration(bar):
    num, den = bar['time']
    return ' '.join(f'r{den}' for _ in range(num))


def _tempo_block(value, denominator):
    """
    Accepts '120' (a number for the denominator's note value) or a full
    '4 = 120'. Anything else becomes a text tempo mark.
    """
    value = value.strip()
    if re.match(r'^\d+$', value):
        return rf'\tempo {denominator} = {value}'
    if re.match(r'^\d+\.?\s*=\s*\d+$', value):
        return r'\tempo %s' % re.sub(r'\s*=\s*', ' = ', value)
    return r'\tempo \markup { "%s" }' % _escape_markup(value)


# Key directives are written as a chord symbol would be: "Bb", "F#m",
# "Eb major", "C minor", "D dorian".
_MODE_WORDS = {
    'major': 'major', 'maj': 'major', 'ionian': 'ionian',
    'minor': 'minor', 'min': 'minor', 'm': 'minor', 'aeolian': 'aeolian',
    'dorian': 'dorian', 'phrygian': 'phrygian', 'lydian': 'lydian',
    'mixolydian': 'mixolydian', 'locrian': 'locrian',
}


def parse_key(value):
    """
    Return (lilypond_pitch, mode) for a key directive, or None.
    'Bb' -> ('bf', 'major'); 'F#m' -> ('fs', 'minor').
    """
    if not value:
        return None
    text = value.strip()
    m = re.match(r'^([A-Ga-g])\s*(bb|##|[b#x])?\s*(.*)$', text)
    if not m:
        return None
    letter = m.group(1).lower()
    accidental = m.group(2) or ''
    rest = m.group(3).strip().lower()

    suffix = {'': '', 'b': 'f', '#': 's', 'bb': 'ff', '##': 'ss', 'x': 'ss'}
    pitch = letter + suffix.get(accidental, '')

    if not rest:
        mode = 'major'
    else:
        mode = _MODE_WORDS.get(rest, _MODE_WORDS.get(rest.split()[0], None))
        if mode is None:
            return None
    return pitch, mode


def _key_block(value, warnings):
    if not value:
        return None
    parsed = parse_key(value)
    if not parsed:
        warnings.append(f"Could not read key: {value}")
        return None
    pitch, mode = parsed
    return rf'\key {pitch} \{mode}'


def chart_chord_names(song):
    """
    Every distinct chord symbol in the changes, in order of first
    appearance. Used to work out which diagrams a chart needs.
    """
    names = []
    for bar in song['bars']:
        for chord in bar['chords']:
            # A '%' bar's chords are copies; the name is already in the
            # list from the bar being repeated.
            if chord.get('silent'):
                continue
            if chord['name'] not in names:
                names.append(chord['name'])
    return names


# ----------------------------------------------------------------------
# Transposition
# ----------------------------------------------------------------------
#
# LilyPond's \transpose moves pitches, and the chord symbols here are
# markup strings -- verified: \transpose c d over this score shifts the
# key signature and leaves "Cm7" reading "Cm7". So transposition happens
# in Python, before the markup is written, and covers three things at
# once: the symbols in the changes, the spellings in the definitions
# (so the diagrams follow), and the key signature.
#
# Only the ROOT of a symbol is parsed -- a letter, its accidentals, and
# an optional slash bass. Everything after the root is carried across
# untouched, so there is still no chord-quality parser and no argument
# about what "alt" or "m7b5" ought to mean.

_LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
_LETTER_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# Root at the start of a chord symbol: letter, then any run of
# accidentals. 'Bbb' is B double-flat, 'C#' is C sharp.
_ROOT_RE = re.compile(r'^([A-G])((?:[b#x]|bb|##)*)')


def _accidental_value(text):
    value = 0
    for ch in text:
        if ch == 'b':
            value -= 1
        elif ch == '#':
            value += 1
        elif ch == 'x':
            value += 2
    return value


def _accidental_text(value):
    if value > 0:
        return '#' * value
    return 'b' * (-value)


def transpose_note(letter, accidental, letter_steps, semitone_steps):
    """
    Move one note by an interval given as (letter steps, semitones).
    Working in letter steps rather than semitones alone is what keeps
    the spelling right: up a minor third from C is Eb, not D#, because
    the letter moves C->D->E regardless of the accidental that lands.
    """
    index = _LETTERS.index(letter)
    new_index = (index + letter_steps) % 7
    new_letter = _LETTERS[new_index]

    octaves = (index + letter_steps) // 7
    old_pc = _LETTER_PC[letter] + accidental
    target_pc = old_pc + semitone_steps
    natural_pc = _LETTER_PC[new_letter] + 12 * octaves
    return new_letter, target_pc - natural_pc


def interval_between(from_key, to_key):
    """
    (letter steps, semitones) from one key to another, as note names.
    Always the smallest upward interval, so 'Bb' to 'C' is up a tone
    rather than down a seventh.
    """
    source = _ROOT_RE.match((from_key or '').strip())
    target = _ROOT_RE.match((to_key or '').strip())
    if not source or not target:
        return None

    from_letter = source.group(1)
    from_acc = _accidental_value(source.group(2))
    to_letter = target.group(1)
    to_acc = _accidental_value(target.group(2))

    letter_steps = (_LETTERS.index(to_letter) - _LETTERS.index(from_letter)) % 7
    from_pc = (_LETTER_PC[from_letter] + from_acc) % 12
    to_pc = (_LETTER_PC[to_letter] + to_acc) % 12
    semitones = (to_pc - from_pc) % 12
    return letter_steps, semitones


def transpose_symbol(symbol, letter_steps, semitone_steps):
    """
    Transpose a chord symbol's root, leaving its quality alone.

    A slash is a bass note only when a note name follows it: 'Dm7/G'
    has a G bass and transposes, while 'C6/9' and 'Gm6/9' are added
    ninths and must not be touched.
    """
    match = _ROOT_RE.match(symbol)
    if not match:
        return symbol

    letter, accidental_text = match.group(1), match.group(2)
    new_letter, new_accidental = transpose_note(
        letter, _accidental_value(accidental_text), letter_steps, semitone_steps
    )
    rest = symbol[match.end():]

    slash = rest.rfind('/')
    if slash != -1:
        bass = rest[slash + 1:]
        bass_match = _ROOT_RE.match(bass)
        if bass_match and bass_match.end() == len(bass):
            bass_letter, bass_new = transpose_note(
                bass_match.group(1), _accidental_value(bass_match.group(2)),
                letter_steps, semitone_steps
            )
            rest = (rest[:slash + 1] + bass_letter + _accidental_text(bass_new))

    return new_letter + _accidental_text(new_accidental) + rest


def transpose_song(song, target_key):
    """
    Transpose a parsed Song in place to target_key, returning an error
    string or None. Needs a source key: without one there is no way to
    tell whether up one semitone should spell sharps or flats.
    """
    if not song.get('key'):
        return ("Transposing needs a starting key -- set the key field or "
                "add a {key: ...} directive.")

    source_root = _ROOT_RE.match(song['key'].strip())
    target_root = _ROOT_RE.match((target_key or '').strip())
    if not source_root:
        return f"Could not read the starting key: {song['key']}"
    if not target_root:
        return f"Could not read the target key: {target_key}"

    interval = interval_between(song['key'], target_key)
    if interval is None:
        return f"Could not work out the interval to {target_key}"
    letter_steps, semitones = interval
    if letter_steps == 0 and semitones == 0:
        return None                       # already in that key

    for bar in song['bars']:
        for chord in bar['chords']:
            chord['name'] = transpose_symbol(chord['name'], letter_steps, semitones)

    new_definitions = {}
    new_order = []
    for name in song['definition_order']:
        definition = song['definitions'][name]
        new_name = transpose_symbol(name, letter_steps, semitones)
        notes = []
        for token in definition['spelling'].split():
            match = _ROOT_RE.match(token)
            if not match:
                notes.append(token)
                continue
            new_letter, new_accidental = transpose_note(
                match.group(1), _accidental_value(match.group(2)),
                letter_steps, semitones
            )
            notes.append(new_letter + _accidental_text(new_accidental))
        new_definitions[new_name] = {
            'spelling': ' '.join(notes),
            'inversions': definition['inversions'],
        }
        new_order.append(new_name)
    song['definitions'] = new_definitions
    song['definition_order'] = new_order

    # The key directive keeps whatever mode word it carried.
    mode_words = song['key'].strip()[source_root.end():].strip()
    song['key'] = (target_key.strip() + (' ' + mode_words if mode_words else ''))
    return None
