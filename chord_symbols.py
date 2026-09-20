"""
Chord symbols: read a written symbol and spell it out.

Turns 'Abm7', 'Ab-7', 'D9', 'D7add9', 'G#7(b9#11)' and the rest of the
usual notations into a list of spelled notes, so a chart can be typed
in the shorthand a player already uses instead of spelling every chord
by hand.

Spelling is done by scale degree, never by pitch class: the 13th of C
is the 6th letter with whatever accidental makes the arithmetic work,
which is why C7b13 comes out C E G Bb Ab and not C E G Bb G#. That is
the same rule the diagram labels use, so the two agree by construction.

Notation this accepts, roughly:

  root        A-G with b / # / bb / ## / x
  major       maj, Maj, MAJ, ma, M, Δ, ^, j
  minor       m, mi, min, -
  diminished  dim, o, °      half-diminished  ø, m7b5
  augmented   aug, +
  suspended   sus, sus2, sus4  (sus with any other number adds it)
  numbers     5 6 7 9 11 13
  alterations b5 #5 b9 #9 #11 b13, and -5 +5 -9 +9 in place of b/#
  additions   add9, add11, add13, add2, add4, add6, 6/9
  omissions   no3, no5, omit3, omit5
  altered     alt
  bass note   /G, /Ab  (a '/' followed by anything else, as in 6/9,
              is part of the quality)

Case matters in exactly one place: bare 'M' is major and bare 'm' is
minor. Everything else is case-insensitive.
"""

import re

LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
LETTER_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# Degree -> (letter steps above the root, semitones in a major scale).
# The compound degrees carry their octave in BOTH halves: a 9th is eight
# letter steps and fourteen semitones, not one step and fourteen, or the
# accidental arithmetic comes out an octave adrift and spells a twelve-
# sharp note.
DEGREE_TABLE = {
    1: (0, 0), 2: (1, 2), 3: (2, 4), 4: (3, 5), 5: (4, 7),
    6: (5, 9), 7: (6, 11), 9: (8, 14), 11: (10, 17), 13: (12, 21),
}

_ROOT_RE = re.compile(r'^([A-Ga-g])((?:bb|##|[b#x])?)')


class ChordSymbolError(ValueError):
    pass


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


def spell_degree(root_letter, root_accidental, degree, alteration=0):
    """
    Spell one degree above a root as (letter, accidental).

    The letter is fixed by the degree and the accidental falls out of
    the arithmetic, so a b13 is always the sixth letter flattened and
    never the fifth letter sharpened.
    """
    letter_steps, semitones = DEGREE_TABLE[degree]
    index = LETTERS.index(root_letter)
    new_letter = LETTERS[(index + letter_steps) % 7]
    octaves = (index + letter_steps) // 7
    target = LETTER_PC[root_letter] + root_accidental + semitones + alteration
    natural = LETTER_PC[new_letter] + 12 * octaves
    return new_letter, target - natural


# ----------------------------------------------------------------------
# Tokenising the quality
# ----------------------------------------------------------------------

_TOKENS = [
    ('sep',    re.compile(r'^[\s,()\[\]]+')),
    ('halfdim', re.compile(r'^(ø|Ø|0)')),
    ('sus',    re.compile(r'^sus(\d+)?', re.I)),
    ('add',    re.compile(r'^add\s*(\d+)', re.I)),
    ('omit',   re.compile(r'^(?:no|omit)\s*(\d+)', re.I)),
    ('alt',    re.compile(r'^alt(?:ered)?', re.I)),
    # 'maj' and friends must be tried before the bare 'm' of minor.
    ('major',  re.compile(r'^(maj|major|ma(?![a-z])|Δ|\^|j(?![a-z]))', re.I)),
    ('major',  re.compile(r'^M(?![a-z])')),          # bare capital M only
    ('minor',  re.compile(r'^(min|mi(?![a-z])|m(?![a-z])|-(?!\d))')),
    ('dim',    re.compile(r'^(dim|°|o(?![a-z]))', re.I)),
    ('aug',    re.compile(r'^(aug|\+(?!\d))', re.I)),
    ('alter',  re.compile(r'^([b#+-])\s*(5|6|9|11|13)')),
    ('number', re.compile(r'^(13|11|9|7|6|5|4|2)')),
    ('slash',  re.compile(r'^/')),
]


def _tokenise(text):
    tokens = []
    position = 0
    while position < len(text):
        for name, pattern in _TOKENS:
            match = pattern.match(text[position:])
            if match:
                if name != 'sep':
                    tokens.append((name, match.groups(), match.group(0)))
                position += match.end()
                break
        else:
            raise ChordSymbolError(
                f"did not understand {text[position:]!r}"
            )
    return tokens


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

def parse_chord_symbol(symbol):
    """
    Parse a chord symbol into

        {'symbol', 'root': (letter, accidental), 'bass': (letter, acc)|None,
         'notes': [(letter, accidental), ...], 'degrees': [int, ...]}

    Notes come out in degree order, low to high, which is the order the
    diagram code expects. Raises ChordSymbolError on anything it cannot
    read, so a caller can fall back rather than draw a wrong chord.
    """
    if not symbol or not symbol.strip():
        raise ChordSymbolError("empty chord symbol")

    text = symbol.strip()
    match = _ROOT_RE.match(text)
    if not match:
        raise ChordSymbolError(f"no root note in {symbol!r}")

    root_letter = match.group(1).upper()
    root_accidental = _accidental_value(match.group(2))
    rest = text[match.end():]

    # A '-' straight after the root means minor ('Ab-7'); later in the
    # symbol it is a flat alteration ('C7-9'). Rewriting it here keeps
    # that distinction out of the token patterns.
    if rest.startswith('-'):
        rest = 'm' + rest[1:]

    # A trailing '/X' is a bass note only when a note name follows; the
    # slash in 6/9 or 7/6 belongs to the quality.
    bass = None
    slash = rest.rfind('/')
    if slash != -1:
        candidate = rest[slash + 1:].strip()
        bass_match = _ROOT_RE.match(candidate)
        if bass_match and bass_match.end() == len(candidate):
            bass = (bass_match.group(1).upper(),
                    _accidental_value(bass_match.group(2)))
            rest = rest[:slash]

    tokens = _tokenise(rest)

    third = 'major'          # major | minor | sus2 | sus4 | None
    fifth = 'perfect'        # perfect | dim | aug | None
    seventh = None           # None | 'b7' | 'maj7' | 'bb7'
    quality_seen = None      # which quality word was written
    sixth = False
    extensions = set()       # 9, 11, 13 present as chord tones
    alterations = {}         # degree -> accidental offset
    omitted = set()
    explicit_number = None

    index = 0
    while index < len(tokens):
        name, groups, raw = tokens[index]
        index += 1

        if name == 'major':
            quality_seen = 'major'
        elif name == 'minor':
            third = 'minor'
            quality_seen = 'minor'
        elif name == 'dim':
            third, fifth = 'minor', 'dim'
            quality_seen = 'dim'
        elif name == 'aug':
            fifth = 'aug'
            quality_seen = 'aug'
        elif name == 'halfdim':
            third, fifth, seventh = 'minor', 'dim', 'b7'
            quality_seen = 'halfdim'
        elif name == 'sus':
            number = int(groups[0]) if groups[0] else 4
            if number == 2:
                third = 'sus2'
            elif number == 4:
                third = 'sus4'
            else:
                # 7sus9 and the like: suspended, with the number read as
                # an extension rather than the suspension itself.
                third = 'sus4'
                _apply_number(number, locals_ref=None,
                              extensions=extensions, seventh_ref=None)
                extensions.add(number)
                if seventh is None:
                    seventh = 'b7'
        elif name == 'add':
            degree = int(groups[0])
            if degree in DEGREE_TABLE:
                extensions.add(degree if degree in (9, 11, 13) else degree)
        elif name == 'omit':
            omitted.add(int(groups[0]))
        elif name == 'alt':
            # Altered dominant. Written 'alt' means "use whichever
            # alterations you like", so this picks the common comping
            # set rather than stacking all four.
            seventh = 'b7'
            fifth = None
            extensions.update({9, 13})
            alterations[9] = 1        # #9
            alterations[13] = -1      # b13
            quality_seen = quality_seen or 'dominant'
        elif name == 'alter':
            sign, degree_text = groups
            degree = int(degree_text)
            offset = 1 if sign in ('#', '+') else -1
            alterations[degree] = offset
            if degree in (9, 11, 13):
                extensions.add(degree)
                if seventh is None and explicit_number is None:
                    seventh = 'b7'
            elif degree == 5:
                fifth = 'alt'
        elif name == 'number':
            number = int(groups[0])
            explicit_number = number
            if number == 5:
                third = None if quality_seen is None else third
                omitted.add(3)
            elif number == 6:
                sixth = True
            elif number == 7:
                seventh = 'maj7' if quality_seen == 'major' else (
                    'bb7' if quality_seen == 'dim' else 'b7')
            elif number in (9, 11, 13):
                # A 6th already written means this is an added tone, not
                # a stacked seventh chord: 6/9 is a sixth chord with a
                # ninth, and has no b7 in it.
                if sixth:
                    pass
                elif quality_seen == 'major':
                    seventh = 'maj7'
                elif quality_seen == 'dim':
                    seventh = 'bb7'
                elif seventh is None:
                    seventh = 'b7'
                extensions.add(9)
                if number >= 11:
                    extensions.add(11)
                if number >= 13:
                    extensions.add(13)
                    # A written 13 implies 9 but conventionally leaves
                    # out the 11, which clashes with the third.
                    if number == 13 and third == 'major':
                        extensions.discard(11)
            elif number in (2, 4):
                extensions.add(9 if number == 2 else 11)
        elif name == 'slash':
            continue

    # 'maj' with no number at all is just a major triad; 'dim' likewise.
    if quality_seen == 'major' and seventh is None and not extensions and not sixth:
        pass

    # ------------------------------------------------------------------
    # Assemble degrees
    # ------------------------------------------------------------------
    degrees = []          # (degree, alteration)

    degrees.append((1, 0))

    if third == 'major' and 3 not in omitted:
        degrees.append((3, 0))
    elif third == 'minor' and 3 not in omitted:
        degrees.append((3, -1))
    elif third == 'sus2':
        degrees.append((2, 0))
    elif third == 'sus4':
        degrees.append((4, 0))

    if fifth and 5 not in omitted:
        if fifth == 'perfect':
            degrees.append((5, 0))
        elif fifth == 'dim':
            degrees.append((5, -1))
        elif fifth == 'aug':
            degrees.append((5, 1))
        elif fifth == 'alt':
            degrees.append((5, alterations.get(5, 0)))

    if sixth:
        degrees.append((6, alterations.get(6, 0)))

    if seventh == 'b7':
        degrees.append((7, -1))
    elif seventh == 'maj7':
        degrees.append((7, 0))
    elif seventh == 'bb7':
        degrees.append((7, -2))

    for degree in (9, 11, 13):
        if degree in extensions:
            degrees.append((degree, alterations.get(degree, 0)))

    # An alteration on a degree that is not otherwise present still
    # sounds -- 'C7b13' has no natural 13 to alter, it has a b13.
    present = {d for d, _ in degrees}
    for degree, offset in sorted(alterations.items()):
        if degree in (9, 11, 13) and degree not in present:
            degrees.append((degree, offset))

    degrees.sort(key=lambda item: (DEGREE_TABLE[item[0]][1] + item[1],
                                   item[0]))

    notes = [
        spell_degree(root_letter, root_accidental, degree, offset)
        for degree, offset in degrees
    ]

    return {
        'symbol': symbol.strip(),
        'root': (root_letter, root_accidental),
        'bass': bass,
        'notes': notes,
        'degrees': [d for d, _ in degrees],
    }


def _apply_number(number, locals_ref, extensions, seventh_ref):
    """Placeholder kept for the sus-with-number path; see caller."""
    return


def spelling_string(parsed):
    """The note list as the 'C E G Bb' text the diagram code takes."""
    return ' '.join(
        letter + _accidental_text(accidental)
        for letter, accidental in parsed['notes']
    )


def try_spelling(symbol):
    """Spelling string for a symbol, or None if it cannot be read."""
    try:
        return spelling_string(parse_chord_symbol(symbol))
    except (ChordSymbolError, KeyError, ValueError):
        return None


# ----------------------------------------------------------------------
# Substitutions
# ----------------------------------------------------------------------
#
# Each entry is (why, letter steps, semitones, suffix): the substitute's
# root is the original root moved by that interval, and the suffix is
# appended to make a new symbol which is then parsed by the code above.
# Interval steps rather than pitch classes again, so a tritone sub of G7
# spells Db7 and not C#7.

_SUB_TABLES = {
    'maj7': [
        ("adds the 6th and 9th, a lighter tonic sound", 0, 0, "6/9"),
        ("the same chord with the 9th on top", 0, 0, "maj9"),
        ("plain 6th chord, no leading tone", 0, 0, "6"),
        ("iii minor: the top four notes of the maj9", 2, 4, "m7"),
        ("vi minor: relative minor, shares three notes", 5, 9, "m7"),
    ],
    'maj6': [
        ("the 6/9 voicing", 0, 0, "6/9"),
        ("major 7th in place of the 6th", 0, 0, "maj7"),
    ],
    'dominant': [
        ("adds the 9th", 0, 0, "9"),
        ("tritone substitution", 4, 6, "7"),
        ("adds the 13th", 0, 0, "13"),
        ("altered 9th for tension", 0, 0, "7b9"),
        ("suspended, delays the third", 0, 0, "7sus4"),
        ("the ii of a ii-V, played in its place", 4, 7, "m7"),
    ],
    'minor7': [
        ("adds the 9th", 0, 0, "m9"),
        ("adds the 11th", 0, 0, "m11"),
        ("minor 6th, brighter than the b7", 0, 0, "m6"),
        ("relative major: shares three notes", 2, 3, "maj7"),
        ("the dominant it implies in a ii-V", 3, 5, "7"),
    ],
    'minor6': [
        ("half-diminished on the 6th, the same four notes", 5, 9, "m7b5"),
        ("minor with a b7 instead", 0, 0, "m7"),
    ],
    'halfdim': [
        ("minor 6th on the b3: the same four notes", 2, 3, "m6"),
        # Up a MINOR SIXTH, not an augmented fifth: Cm7b5 is the top of
        # Ab9, and spelling it G#9 would be the same keys under a name
        # nobody writes.
        ("the dominant it is the rootless top of", 5, 8, "9"),
    ],
    'dim7': [
        ("the dominant b9 it is the rootless top of", 5, 8, "7b9"),
        ("half-diminished, one note softer", 0, 0, "m7b5"),
    ],
    'minor': [
        ("adds the b7", 0, 0, "m7"),
        ("adds the 6th", 0, 0, "m6"),
    ],
    'major': [
        ("adds the major 7th", 0, 0, "maj7"),
        ("adds the 6th", 0, 0, "6"),
    ],
}


def _family(parsed):
    """Which substitution table applies to a parsed chord."""
    degrees = dict()
    for degree, offset in zip(parsed['degrees'], _offsets(parsed)):
        degrees[degree] = offset

    third = degrees.get(3)
    fifth = degrees.get(5)
    seventh = degrees.get(7)
    has_six = 6 in degrees

    if seventh == -2:
        return 'dim7'
    if third == -1 and fifth == -1 and seventh == -1:
        return 'halfdim'
    if third == -1:
        if has_six and seventh is None:
            return 'minor6'
        if seventh == -1:
            return 'minor7'
        if seventh is None:
            return 'minor'
        return 'minor7'
    if third == 0 or third is None:
        if seventh == 0:
            return 'maj7'
        if seventh == -1:
            return 'dominant'
        if has_six:
            return 'maj6'
        return 'major'
    return None


def _offsets(parsed):
    """Recover each degree's alteration from the spelled notes."""
    root_letter, root_accidental = parsed['root']
    out = []
    for degree, (letter, accidental) in zip(parsed['degrees'], parsed['notes']):
        natural_letter, natural_accidental = spell_degree(
            root_letter, root_accidental, degree, 0)
        out.append(accidental - natural_accidental)
    return out


def _spell_at(root_letter, root_accidental, letter_steps, semitones):
    index = LETTERS.index(root_letter)
    new_letter = LETTERS[(index + letter_steps) % 7]
    octaves = (index + letter_steps) // 7
    target = LETTER_PC[root_letter] + root_accidental + semitones
    natural = LETTER_PC[new_letter] + 12 * octaves
    return new_letter, target - natural


def _transposed_root(root_letter, root_accidental, letter_steps, semitones):
    """
    Root of a substitute chord, spelled the way a player would write it.

    A tritone is an augmented fourth or a diminished fifth depending on
    where you start: G7's tritone sub is Db7 (four letter steps), while
    Db7's is G7 (three). Picking one fixed interval spells the other as
    Abb7. So the neighbouring letter is tried whenever the obvious one
    needs a double accidental, and the simpler spelling wins.
    """
    letter, accidental = _spell_at(
        root_letter, root_accidental, letter_steps, semitones)
    if abs(accidental) < 2:
        return letter, accidental

    best = (letter, accidental)
    for alternative in (letter_steps - 1, letter_steps + 1):
        candidate = _spell_at(
            root_letter, root_accidental, alternative, semitones)
        if abs(candidate[1]) < abs(best[1]):
            best = candidate
    return best


def substitutions(symbol, limit=3):
    """
    Common substitutes for a chord symbol, as
    [{'symbol', 'why', 'spelling'}, ...], most idiomatic first.

    Returns [] for a symbol that cannot be read or has no table. Each
    substitute is generated as a symbol and then parsed by the same
    parser, so anything suggested here is something the rest of the app
    can already spell and draw.
    """
    try:
        parsed = parse_chord_symbol(symbol)
    except (ChordSymbolError, KeyError, ValueError):
        return []

    table = _SUB_TABLES.get(_family(parsed))
    if not table:
        return []

    root_letter, root_accidental = parsed['root']
    out = []
    for why, letter_steps, semitones, suffix in table:
        letter, accidental = _transposed_root(
            root_letter, root_accidental, letter_steps, semitones)
        candidate = letter + _accidental_text(accidental) + suffix
        if candidate == parsed['symbol']:
            continue
        spelling = try_spelling(candidate)
        if not spelling:
            continue
        out.append({'symbol': candidate, 'why': why, 'spelling': spelling})
        if len(out) >= limit:
            break
    return out
