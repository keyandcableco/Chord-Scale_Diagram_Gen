#!/usr/bin/env python3
"""
app.py -- local web GUI for chord_diagram.py

Run with:
    python3 app.py

Then open http://127.0.0.1:5000 in a browser.

This is a thin orchestration layer: it collects the same decisions the
command-line script asks for via input() -- chord vs scale, spelling,
how many inversions, whether to show the full chord, whether to
harmonize a scale and render its chords too -- as one web form, then
calls the exact same underlying functions from chord_diagram.py
(parse_note, render_chord_with_inversions, harmonize,
draw_harmonized_chart, etc.) to do the actual work. Nothing about the
music theory or rendering logic is reimplemented here.

Each request runs in its own temporary working directory so concurrent
users (or repeated runs) never collide on filenames, and the directory
is cleaned up after the response is sent.
"""

import os
import shutil
import tempfile
import traceback
import uuid

from flask import Flask, request, render_template, send_file, jsonify, after_this_request

import chord_diagram as cd
import leadsheet
import chord_symbols

app = Flask(__name__)

# Where finished runs get parked briefly so send_file can serve them
# after the per-request temp dir would otherwise already be gone.
RESULTS_DIR = os.path.join(tempfile.gettempdir(), "chord_diagram_webapp_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline(form, work_dir):
    """
    form: werkzeug form dict from the POST request.
    work_dir: directory to generate all files in (already created,
    already the current working directory by the time this is called).

    Returns (success: bool, message: str, output_path: str or None,
    output_is_zip: bool).
    """
    mode = form.get("mode", "chord")
    name = (form.get("name") or "").strip() or ("Untitled chord" if mode == "chord" else "Untitled scale")
    slug = cd.slugify(name)

    sub_entries = []  # (sub_name, ly_blocks, keyboard_images, extra_score_blocks)
    key_signature_note = None

    if mode == "chord":
        spelling = form.get("chord_spelling", "")
        tokens = spelling.split()
        if not tokens:
            return False, "Please enter a chord spelling.", None, False
        parsed_root = [cd.parse_note(t) for t in tokens]

        inversions_raw = form.get("inversions", "").strip()
        how_many = int(inversions_raw) if inversions_raw else len(parsed_root)
        want_full_chord = form.get("full_chord") == "on"
        want_info_panel = form.get("info_panel") == "on"
        want_guitar_shapes = form.get("guitar_shapes") == "on"
        want_jazz_shapes = form.get("jazz_shapes") == "on"
        want_pedal = form.get("pedal_voicings") == "on"
        rootless_gtr = form.get("rootless_guitar") == "on"
        rootless_kbd = form.get("rootless_piano") == "on"
        want_full_fretboard = form.get("full_fretboard") == "on"

        images, blocks = cd.render_chord_with_inversions(
            parsed_root, name, slug,
            how_many=how_many, want_full_chord=want_full_chord,
            want_info_panel=want_info_panel, want_guitar_shapes=want_guitar_shapes,
            want_jazz_shapes=want_jazz_shapes, want_full_fretboard=want_full_fretboard,
            want_pedal_voicings=want_pedal,
            rootless_guitar=rootless_gtr, rootless_keyboard=rootless_kbd,
        )
        sub_entries.append((name, blocks, images, []))

    else:  # scale
        spelling = form.get("scale_spelling", "")
        tokens = spelling.split()
        if not tokens:
            return False, "Please enter a scale spelling.", None, False
        parsed = [cd.parse_note(t) for t in tokens]

        notes_with_octaves = cd.assign_octaves(parsed, start_octave=4, mode='scale')
        root_letter, root_accidental = parsed[0][0], parsed[0][1]
        degree_labels = cd.degree_labels_for_voicing(notes_with_octaves, root_letter, root_accidental)

        cd.draw_keyboard(notes_with_octaves, degree_labels, slug)
        scale_images = [(name, slug)]
        scale_blocks = [cd.ly_scale_block(name, notes_with_octaves)]

        if form.get("scale_full_fretboard") == "on":
            scale_images.extend(cd.draw_full_fretboard_scale(parsed, name, slug))

        # Drop a trailing repeated tonic if present, same as the CLI.
        harmonize_notes = parsed
        if len(parsed) > 1:
            first_pc = (cd.PITCH_CLASS[parsed[0][0]] + parsed[0][1]) % 12
            last_pc = (cd.PITCH_CLASS[parsed[-1][0]] + parsed[-1][1]) % 12
            if first_pc == last_pc:
                harmonize_notes = parsed[:-1]

        if form.get("scale_full_keyboard") == "on":
            full_scale_voicing = cd.generate_full_chord_voicing(
                harmonize_notes, cd.FULL_SCALE_KEYBOARD_OCTAVES, start_octave=3
            )
            full_scale_labels = cd.degree_labels_for_voicing(
                full_scale_voicing, root_letter, root_accidental,
                extension_pcs=set(),
            )
            full_scale_img_base = f"{slug}-full-scale"
            cd.draw_keyboard(full_scale_voicing, full_scale_labels, full_scale_img_base,
                              min_octaves=cd.FULL_SCALE_KEYBOARD_OCTAVES)
            scale_images.append((f"{name} \u2013 Full scale", full_scale_img_base))

        # Key signature staff -- a small extra \score (no notes, just
        # clef + key signature) appended after the main one. Only
        # meaningful for a 7-note scale.
        extra_score_blocks = []
        key_signature_note = None
        if form.get("key_signature") == "on":
            ks_score_block, ks_message = cd.key_signature_block(harmonize_notes)
            if ks_score_block is not None:
                extra_score_blocks.append(ks_score_block)
            key_signature_note = ks_message

        want_chart = form.get("harmonize") == "on" and len(harmonize_notes) >= 3
        if want_chart:
            chord_size = 4 if (form.get("chord_size") == "7" and len(harmonize_notes) >= 4) else 3

            chart_suffix = "harmonized-7ths" if chord_size == 4 else "harmonized"
            chart_label_suffix = "Harmonized 7th chords" if chord_size == 4 else "Harmonized chords"
            chart_base = f"{slug}-{chart_suffix}"
            harmonized_rows = cd.harmonize(harmonize_notes, chord_size=chord_size)
            cd.draw_harmonized_chart(harmonize_notes, name, chart_base, chord_size=chord_size)
            scale_images.append((f"{name} \u2013 {chart_label_suffix}", chart_base))

            want_render_chords = form.get("render_chords") == "on"
            if want_render_chords:
                chord_how_many = form.get("chord_inversions", "")
                chord_full_chord = form.get("chord_full_chord") == "on"
                chord_info_panel = form.get("chord_info_panel") == "on"
                chord_guitar_shapes = form.get("chord_guitar_shapes") == "on"
                chord_jazz_shapes = form.get("chord_jazz_shapes") == "on"
                chord_full_fretboard = form.get("chord_full_fretboard") == "on"

                for row in harmonized_rows:
                    chord_root_display = row["notes"][0]
                    quality_word = cd.QUALITY_WORD.get(
                        row["quality_abbrev"], row["quality_name"].lower()
                    )
                    chord_name = f"{chord_root_display} {quality_word}"
                    full_chord_label = f"{chord_name} ({row['numeral']} of {name})"
                    chord_slug = f"{slug}-{cd.slugify(row['numeral'])}-{cd.slugify(chord_root_display)}"

                    chord_root_notes = [cd.parse_note(n) for n in row["notes"]]

                    this_how_many = (
                        int(chord_how_many) if chord_how_many
                        else len(chord_root_notes)
                    )
                    images, blocks = cd.render_chord_with_inversions(
                        chord_root_notes, full_chord_label, chord_slug,
                        how_many=this_how_many, want_full_chord=chord_full_chord,
                        want_info_panel=chord_info_panel,
                        want_guitar_shapes=chord_guitar_shapes,
                        want_jazz_shapes=chord_jazz_shapes,
                        want_full_fretboard=chord_full_fretboard,
                    )
                    sub_entries.append((full_chord_label, blocks, images, []))

        sub_entries.insert(0, (name, scale_blocks, scale_images, extra_score_blocks))

    # Write one .ly per sub-entry (own staff each), matching the CLI's
    # per-chord-gets-its-own-page structure.
    entries_data = []
    all_keyboard_images = []
    for sub_name, ly_blocks, keyboard_images, extra_score_blocks in sub_entries:
        sub_slug = cd.slugify(sub_name)
        ly_filename = f"{sub_slug}.ly"
        if any(existing == ly_filename for _, existing, _ in entries_data):
            ly_filename = f"{sub_slug}-{uuid.uuid4().hex[:6]}.ly"
        with open(ly_filename, "w") as f:
            f.write(cd.build_single_lilypond(sub_name, ly_blocks, extra_score_blocks))
        entries_data.append((sub_name, ly_filename, keyboard_images))
        all_keyboard_images.extend(keyboard_images)

    doc_title = entries_data[0][0] if len(entries_data) == 1 else "Chord and Scale Reference"
    output_stem = cd.slugify(form.get("output_filename") or doc_title)

    image_basenames = {base for _, base in all_keyboard_images}
    if output_stem in image_basenames:
        output_stem = f"{output_stem}-doc"

    lytex_filename = f"{output_stem}.lytex"
    with open(lytex_filename, "w") as f:
        f.write(cd.build_lytex_wrapper(doc_title, entries_data))

    success, message = cd.try_run_lilypond_book(lytex_filename, all_keyboard_images)

    if mode == "scale" and key_signature_note:
        message = f"{key_signature_note}\n\n{message}"

    if success:
        out_dir = "lilypond-book-out"
        final_pdf = lytex_filename.replace(".lytex", ".pdf")
        pdf_path = os.path.join(work_dir, out_dir, final_pdf)
        return True, message, pdf_path, False
    else:
        # LilyPond/pdflatex isn't available in this environment -- give
        # back a zip of everything that WAS generated (the .ly files,
        # the .lytex wrapper, and every keyboard/chart PDF+SVG) so the
        # person can still get useful output and finish the PDF
        # themselves once those tools are installed.
        #
        # IMPORTANT: the zip must be built OUTSIDE work_dir. Writing it
        # inside the same directory shutil.make_archive is about to zip
        # means the (still being written) zip file gets swept into its
        # own archive, producing a corrupt/self-nesting result.
        zip_base = os.path.join(tempfile.gettempdir(), f"{output_stem}-{uuid.uuid4().hex[:8]}")
        zip_path = shutil.make_archive(zip_base, "zip", work_dir)
        return False, message, zip_path, True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This is a local, single-user tool (one person, one browser tab,
    # one request at a time) -- so a plain os.chdir() per request is
    # fine here. It would NOT be safe if this were ever exposed to
    # multiple concurrent visitors, since chdir affects the whole
    # process; see the project README for notes if that ever changes.
    request_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(prefix=f"chord_{request_id}_")
    prev_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        success, message, output_path, is_zip = run_pipeline(request.form, work_dir)
    except Exception as e:
        os.chdir(prev_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        os.chdir(prev_cwd)

    if output_path is None:
        shutil.rmtree(work_dir, ignore_errors=True)
        return jsonify({"ok": False, "error": message}), 400

    # Move the result somewhere stable, then clean up the working
    # directory -- send_file needs the file to still exist when Flask
    # actually streams it, which can be after this function returns.
    result_name = f"{request_id}{'.zip' if is_zip else '.pdf'}"
    result_path = os.path.join(RESULTS_DIR, result_name)
    shutil.move(output_path, result_path)

    @after_this_request
    def cleanup(response):
        shutil.rmtree(work_dir, ignore_errors=True)
        return response

    # The on-disk temp zip filename may carry a uuid suffix to avoid
    # collisions (see run_pipeline), but the name offered to the person
    # downloading it should stay clean -- derive it from the output
    # stem the same way run_pipeline named the document, not from
    # output_path's actual (possibly uuid-suffixed) basename.
    clean_stem = os.path.basename(output_path)
    if is_zip:
        # strip a trailing "-<8 hex chars>" uuid suffix if present
        import re as _re
        clean_stem = _re.sub(r'-[0-9a-f]{8}\.zip$', '.zip', clean_stem)
    download_name = clean_stem

    return jsonify({
        "ok": success,
        "message": message,
        "download_url": f"/download/{result_name}/{download_name}",
        "is_zip": is_zip,
    })


@app.route("/download/<result_name>/<download_name>")
def download(result_name, download_name):
    result_path = os.path.join(RESULTS_DIR, result_name)
    if not os.path.exists(result_path):
        return "File no longer available -- please generate again.", 404

    @after_this_request
    def cleanup(response):
        try:
            os.remove(result_path)
        except OSError:
            pass
        return response

    return send_file(result_path, as_attachment=True, download_name=download_name)


@app.route("/view/<result_name>")
def view_pdf(result_name):
    """Serve a generated PDF inline (no attachment header) so it renders
    in an iframe or browser PDF viewer without triggering a download.
    Does NOT delete the file after serving -- the download route does that."""
    # Basic safety check: result_name must be a plain filename, no path traversal.
    if "/" in result_name or "\\" in result_name or ".." in result_name:
        return "Invalid filename.", 400
    result_path = os.path.join(RESULTS_DIR, result_name)
    if not os.path.exists(result_path):
        return "File no longer available -- please generate again.", 404
    return send_file(result_path, as_attachment=False, mimetype="application/pdf")


@app.route("/song")
def song():
    return render_template("song.html")


@app.route("/notation")
def notation():
    return render_template("notation.html")


@app.route("/blanks")
def blanks():
    return render_template("blanks.html")


@app.route("/generate_blanks", methods=["POST"])
def generate_blanks():
    request_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(prefix=f"blanks_{request_id}_")
    prev_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        result = _run_blanks_pipeline(request.form, work_dir)
        os.chdir(prev_cwd)

        success, message, output_path, is_zip = result

        if output_path is None:
            shutil.rmtree(work_dir, ignore_errors=True)
            return jsonify({"ok": False, "error": message}), 400

        result_ext = ".zip" if is_zip else ".pdf"
        result_name = f"{request_id}{result_ext}"
        result_path = os.path.join(RESULTS_DIR, result_name)
        shutil.move(output_path, result_path)

        @after_this_request
        def cleanup(response):
            shutil.rmtree(work_dir, ignore_errors=True)
            return response

        download_name = os.path.basename(output_path)

        return jsonify({
            "ok": success,
            "message": message,
            "download_url": f"/download/{result_name}/{download_name}",
            "view_url": f"/view/{result_name}" if not is_zip else None,
            "is_zip": is_zip,
        })

    except Exception as e:
        os.chdir(prev_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400


def _run_blanks_pipeline(form, work_dir):
    title          = (form.get("title") or "").strip()
    want_keyboard  = form.get("keyboard") == "on"
    keyboard_oct   = max(1, min(5, int(form.get("keyboard_octaves") or 2)))
    keyboard_lbls  = form.get("keyboard_labels") == "on"
    want_chord_box = form.get("chord_boxes") == "on"
    chord_cols     = max(1, min(6, int(form.get("chord_box_cols") or 4)))
    chord_rows     = max(1, min(8, int(form.get("chord_box_rows") or 3)))
    want_full_fret = form.get("full_fretboard") == "on"
    want_harmony   = form.get("harmony_grid") == "on"
    harmony_rows   = max(1, min(12, int(form.get("harmony_rows") or 7)))
    harmony_octs   = max(1, min(2,  int(form.get("harmony_octaves") or 2)))
    harmony_names  = form.get("harmony_interval_names") == "on"
    want_staves    = form.get("staves") == "on"
    stave_systems  = max(1, min(20, int(form.get("stave_systems") or 8)))
    stave_bars     = max(1, min(8,  int(form.get("stave_bars") or 4)))
    stave_grand    = form.get("stave_grand") == "on"
    stave_time     = (form.get("stave_time") or "").strip()
    stave_size     = max(10, min(40, int(form.get("stave_size") or 20)))

    if not any([want_keyboard, want_chord_box, want_full_fret,
                want_harmony, want_staves]):
        return False, "Please select at least one template to generate.", None, False

    lytex_src, all_images = cd.build_blank_templates_lytex(
        title or "Blank Templates",
        want_keyboard=want_keyboard,
        keyboard_octaves=keyboard_oct,
        keyboard_labels=keyboard_lbls,
        want_chord_boxes=want_chord_box,
        chord_box_cols=chord_cols,
        chord_box_rows=chord_rows,
        want_full_fretboard=want_full_fret,
        want_harmony_grid=want_harmony,
        harmony_rows=harmony_rows,
        harmony_octaves=harmony_octs,
        harmony_interval_names=harmony_names,
        harmony_label=title,
        want_staves=want_staves,
        stave_systems=stave_systems,
        stave_bars=stave_bars,
        stave_grand=stave_grand,
        stave_time=stave_time,
        stave_size=stave_size,
        work_dir=work_dir,
    )

    output_stem = cd.slugify(title or "blank-templates")
    lytex_file = f"{output_stem}-blanks.lytex"
    with open(lytex_file, "w") as f:
        f.write(lytex_src)

    import shutil as _shutil
    if _shutil.which("lilypond-book") or os.path.isfile("/usr/bin/lilypond-book"):
        out_dir = f"{output_stem}-lb-out"
        success, message = cd.try_run_lilypond_book(
            lytex_file, all_images, out_dir=out_dir
        )
        if success:
            pdf_name = lytex_file.replace(".lytex", ".pdf")
            pdf_path = os.path.join(work_dir, out_dir, pdf_name)
            return True, message, pdf_path, False
        # lilypond-book ran but failed -- fall through to pdflatex,
        # but keep the error so we can show it alongside the result.
        lb_error = message
    else:
        lb_error = "lilypond-book not found"

    # pdflatex only (no staves if lilypond-book missing)
    tex_file = f"{output_stem}-blanks.tex"
    no_stave_src, _ = cd.build_blank_templates_lytex(
        title or "Blank Templates",
        want_keyboard=want_keyboard, keyboard_octaves=keyboard_oct,
        keyboard_labels=keyboard_lbls,
        want_chord_boxes=want_chord_box, chord_box_cols=chord_cols,
        chord_box_rows=chord_rows,
        want_full_fretboard=want_full_fret,
        want_harmony_grid=want_harmony,
        harmony_rows=harmony_rows,
        harmony_octaves=harmony_octs,
        harmony_interval_names=harmony_names,
        harmony_label=title,
        want_staves=False,   # skip staves without lilypond-book
        stave_systems=stave_systems, stave_bars=stave_bars,
        stave_grand=stave_grand, stave_time=stave_time, stave_size=stave_size,
        work_dir=work_dir,
    )
    with open(tex_file, "w") as f:
        f.write(no_stave_src.replace(r"\begin{lilypond}", "% lilypond-book unavailable")
                            .replace(r"\end{lilypond}", ""))
    success, message, pdf_path = cd.try_run_pdflatex(tex_file)
    if success:
        # try_run_pdflatex returns a relative path; make it absolute
        # before we chdir back so shutil.move can find the file.
        pdf_path = os.path.join(work_dir, pdf_path)
        return True, message + f" (staves omitted -- {lb_error})", pdf_path, False

    zip_base = os.path.join(tempfile.gettempdir(),
                            f"{output_stem}-{uuid.uuid4().hex[:8]}")
    zip_path = shutil.make_archive(zip_base, "zip", work_dir)
    return False, message, zip_path, True


@app.route("/generate_notation", methods=["POST"])
def generate_notation():
    """
    Generate a lesson/notation sheet: a LilyPond-engraved staff from
    the user's raw music input, wrapped with title/composer/key/clef
    metadata, followed by optional chord reference diagrams.
    """
    request_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(prefix=f"notation_{request_id}_")
    prev_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        result = _run_notation_pipeline(request.form, work_dir)
    except Exception as e:
        os.chdir(prev_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        os.chdir(prev_cwd)

    success, message, output_path, is_zip = result

    if output_path is None:
        shutil.rmtree(work_dir, ignore_errors=True)
        return jsonify({"ok": False, "error": message}), 400

    result_ext = ".zip" if is_zip else ".pdf"
    result_name = f"{request_id}{result_ext}"
    result_path = os.path.join(RESULTS_DIR, result_name)
    shutil.move(output_path, result_path)

    @after_this_request
    def cleanup(response):
        shutil.rmtree(work_dir, ignore_errors=True)
        return response

    download_name = os.path.basename(output_path)

    return jsonify({
        "ok": success,
        "message": message,
        "download_url": f"/download/{result_name}/{download_name}",
        "view_url": f"/view/{result_name}" if not is_zip else None,
        "is_zip": is_zip,
    })


def _parse_chord_list(chord_list_raw):
    """
    Parse a chord list where each line has the format:
        Name: C E G B
        Name: C E G B : 0 1 2
    The optional third field (after the second colon) is a
    space-separated list of 0-based inversion indices to render.
    Returns (entries, errors) where entries is a list of dicts with
    keys 'name', 'parsed', 'inversions'.
    """
    entries = []
    errors  = []
    for lineno, line in enumerate(chord_list_raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            errors.append(
                f"Line {lineno}: missing colon -- expected 'Name: spelling'"
            )
            continue
        # Split on colons: first = name, second = spelling, optional third = inversions
        parts = line.split(":")
        name          = parts[0].strip()
        spelling_part = parts[1].strip() if len(parts) > 1 else ""
        inv_part      = parts[2].strip() if len(parts) > 2 else ""

        # Parse inversion indices -- any non-integer tokens are ignored.
        inversions = []
        if inv_part:
            for tok in inv_part.split():
                try:
                    inversions.append(int(tok))
                except ValueError:
                    pass
        if not inversions:
            inversions = [0]  # default: root position only

        tokens = spelling_part.split()
        if not tokens:
            errors.append(f"Line {lineno}: no notes after the colon")
            continue
        try:
            parsed = [cd.parse_note(t) for t in tokens]
        except ValueError as e:
            errors.append(f"Line {lineno} ({name}): {e}")
            continue
        entries.append({"name": name, "parsed": parsed, "inversions": inversions})
    return entries, errors


def _run_notation_pipeline(form, work_dir):
    title        = (form.get("title")       or "").strip()
    composer     = (form.get("composer")    or "").strip()
    key_tonic_raw = (form.get("key_tonic") or "").strip()
    key_mode     = (form.get("key_mode")    or "major").strip().lower()
    clef         = (form.get("clef")        or "treble").strip().lower()
    grand_staff  = form.get("grand_staff") == "on"
    treble_music = (form.get("treble_music") or "").strip()
    bass_music   = (form.get("bass_music")   or "").strip()
    want_key_sig = form.get("key_sig") == "on"
    want_jazz    = form.get("guitar_jazz") == "on"
    want_pedal   = form.get("pedal_voicings") == "on"
    want_caged   = form.get("guitar_caged") == "on"
    want_piano   = form.get("piano") == "on"
    show_note_names = form.get("show_note_names") == "on"
    optional_fifth = form.get("optional_fifth") == "on"
    want_scale_kb  = form.get("scale_keyboard") == "on"
    want_scale_fret= form.get("scale_fretboard") == "on"
    chord_list_raw = (form.get("chord_list") or "").strip()
    # Time signature: from selector or free-text override field.
    time_sig_select = (form.get("time_sig_select") or "").strip()
    time_sig_custom = (form.get("time_sig_custom") or "").strip()
    time_sig = time_sig_custom if time_sig_custom else (
        time_sig_select if time_sig_select != "free" else None
    )
    use_relative = form.get("use_relative") == "on"
    try:
        staff_size = max(10, min(50, int(form.get("staff_size") or 26)))
    except (ValueError, TypeError):
        staff_size = 26

    if not treble_music:
        return False, "Please enter some LilyPond music in the notation field.", None, False

    # Parse key tonic.
    key_tonic_letter = key_tonic_accidental = None
    key_score_block = None
    if key_tonic_raw:
        try:
            tonic_parsed = cd.parse_note(key_tonic_raw.split()[0])
            key_tonic_letter     = tonic_parsed[0]
            key_tonic_accidental = tonic_parsed[1]
            if want_key_sig:
                key_score_block = cd.key_signature_score_for_key(
                    key_tonic_letter, key_tonic_accidental, key_mode
                )
        except (ValueError, IndexError):
            pass

    # Key scale diagrams (keyboard and/or guitar fretboard)
    key_scale_images = []
    if key_tonic_letter and (want_scale_kb or want_scale_fret):
        slug_prefix = cd.slugify(f"{key_tonic_raw} {key_mode}")
        key_scale_images = cd.draw_key_scale_diagrams(
            key_tonic_letter, key_tonic_accidental, key_mode,
            slug_prefix,
            want_keyboard=want_scale_kb,
            want_fretboard=want_scale_fret,
        )

    # Parse optional chord list.
    chord_entries = []
    all_images = list(key_scale_images)
    if chord_list_raw:
        chord_entries, _ = _parse_chord_list(chord_list_raw)
        for entry in chord_entries:
            name       = entry["name"]
            parsed     = entry["parsed"]
            inversions = entry["inversions"]
            slug       = cd.slugify(name)
            entry["guitar_images"]   = []
            entry["keyboard_images"] = []

            if want_jazz:
                jazz_imgs = cd.draw_jazz_shapes(
                    parsed, name, slug,
                    optional_fifth=optional_fifth,
                    max_voicings=1,
                    include_open_pedal=want_pedal,
                )
                entry["guitar_images"].extend(jazz_imgs)
                all_images.extend(jazz_imgs)

            if want_caged:
                root_letter     = parsed[0][0]
                root_accidental = parsed[0][1]
                root_pc         = (cd.PITCH_CLASS[root_letter] + root_accidental) % 12
                third_pc = (cd.PITCH_CLASS[parsed[1][0]] + parsed[1][1]) % 12 \
                    if len(parsed) > 1 else (root_pc + 4) % 12
                fifth_pc = (cd.PITCH_CLASS[parsed[2][0]] + parsed[2][1]) % 12 \
                    if len(parsed) > 2 else (root_pc + 7) % 12
                triad_q = cd.classify_triad(root_pc, third_pc, fifth_pc)[1]
                if triad_q in ('maj', 'min'):
                    caged_imgs = cd.draw_all_caged_shapes(
                        root_letter, root_accidental, triad_q, name, slug
                    )
                    entry["guitar_images"].extend(caged_imgs)
                    all_images.extend(caged_imgs)

            if want_piano:
                kb_imgs = cd.draw_keyboard_inversions(parsed, name, slug, inversions)
                entry["keyboard_images"].extend(kb_imgs)
                all_images.extend(kb_imgs)

    # Build .lytex and compile.
    output_stem = cd.slugify(title or "notation-sheet")
    lytex_filename = f"{output_stem}-notation.lytex"
    tex_source = cd.build_notation_sheet_lytex(
        title, composer,
        key_tonic_letter, key_tonic_accidental, key_mode,
        clef, grand_staff,
        treble_music, bass_music,
        key_score_block,
        chord_entries,
        time_sig=time_sig,
        use_relative=use_relative,
        key_scale_images=key_scale_images,
        show_note_names=show_note_names,
        staff_size=staff_size,
    )
    with open(lytex_filename, "w") as f:
        f.write(tex_source)

    import shutil as _shutil
    if _shutil.which("lilypond-book") or os.path.isfile("/usr/bin/lilypond-book"):
        out_dir = f"{output_stem}-lb-out"
        success, message = cd.try_run_lilypond_book(
            lytex_filename, all_images, out_dir=out_dir
        )
        if success:
            tex_stem = lytex_filename.replace(".lytex", "")
            pdf_path = os.path.join(work_dir, out_dir, f"{tex_stem}.pdf")
            return True, message, pdf_path, False

    # Fallback: zip the assets.
    zip_base = os.path.join(
        tempfile.gettempdir(),
        f"{output_stem}-{uuid.uuid4().hex[:8]}"
    )
    zip_path = shutil.make_archive(zip_base, "zip", work_dir)
    msg = "lilypond-book not found -- returning source files."
    return False, msg, zip_path, True


@app.route("/generate_song", methods=["POST"])
def generate_song():
    """
    Parse a chord list of the form:
        Cmaj7: C E G B
        Fm11: F Ab Eb Bb
        ...
    and generate a compact grid PDF with guitar and/or keyboard
    diagrams for each chord, all on as few pages as possible.
    """
    request_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(prefix=f"song_{request_id}_")
    prev_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        result = _run_song_pipeline(request.form, work_dir)
    except Exception as e:
        os.chdir(prev_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        os.chdir(prev_cwd)

    success, message, output_path, is_zip = result

    if output_path is None:
        shutil.rmtree(work_dir, ignore_errors=True)
        return jsonify({"ok": False, "error": message}), 400

    result_ext = ".zip" if is_zip else ".pdf"
    result_name = f"{request_id}{result_ext}"
    result_path = os.path.join(RESULTS_DIR, result_name)
    shutil.move(output_path, result_path)

    @after_this_request
    def cleanup(response):
        shutil.rmtree(work_dir, ignore_errors=True)
        return response

    download_name = os.path.basename(output_path)

    return jsonify({
        "ok": success,
        "message": message,
        "download_url": f"/download/{result_name}/{download_name}",
        # Inline preview URL for the iframe on the song page. /view does
        # not delete the file, so previewing and then downloading both
        # work; /download is still what cleans it up. A zip has nothing
        # to preview.
        "view_url": f"/view/{result_name}" if not is_zip else None,
        "is_zip": is_zip,
    })


def _entry_from_spelling(name, spelling, inversions, errors):
    """Build one chord entry, or record why it could not be built."""
    tokens = spelling.split()
    if not tokens:
        errors.append(f"{name}: no notes after the colon")
        return None
    try:
        parsed = [cd.parse_note(t) for t in tokens]
    except ValueError as e:
        errors.append(f"{name}: {e}")
        return None
    return {"name": name, "parsed": parsed, "inversions": inversions or [0]}


def _parse_chord_list_from_song(song, want_subs=False, sub_limit=2):
    """
    Turn a parsed Song into the chord entries the diagram generators
    expect.

    Three sources, in priority order: a written definition wins, then
    the chord-symbol parser reads the symbol itself, and anything
    neither can handle is reported. That means a chart of bare symbols
    now produces diagrams without a definition in sight, while a
    definition still overrides the parser for a chord you want voiced
    your own way.

    Chart order wins when there are bars, since that is the order the
    tune is read in; a definitions-only list keeps its written order,
    as it always did.
    """
    entries = []
    errors = []
    warnings = []
    seen = set()

    def add(name, spelling, inversions):
        if name in seen:
            return
        entry = _entry_from_spelling(name, spelling, inversions, errors)
        if entry:
            entries.append(entry)
            seen.add(name)

    order = leadsheet.chart_chord_names(song) if song["bars"] else []
    for name in order + [n for n in song["definition_order"] if n not in order]:
        definition = song["definitions"].get(name)
        if definition:
            add(name, definition["spelling"], definition["inversions"])
            continue
        spelling = chord_symbols.try_spelling(name)
        if spelling:
            add(name, spelling, None)
        else:
            warnings.append(
                f"Could not work out the notes of '{name}' -- it will "
                "print on the chart but gets no diagram. Add a "
                f"'{name}: <notes>' line to spell it out."
            )

    if want_subs:
        # Substitutes are appended after the chords they stand in for,
        # so the sheet reads original-then-alternatives rather than
        # interleaving two different harmonies.
        with_subs = []
        for entry in entries:
            with_subs.append(entry)
            for sub in chord_symbols.substitutions(entry["name"],
                                                   limit=sub_limit):
                label = f"{entry['name']} sub: {sub['symbol']}"
                sub_entry = _entry_from_spelling(
                    label, sub["spelling"], None, errors)
                if sub_entry:
                    with_subs.append(sub_entry)
        entries = with_subs

    return entries, errors, warnings


def _run_song_pipeline(form, work_dir):
    """
    Core logic for the song sheet generator. Parses the chord list,
    generates diagram images for each chord, assembles a .lytex
    document with a LilyPond key-signature staff, title, and composer,
    and compiles it with lilypond-book + pdflatex.

    Falls back to a plain pdflatex .tex (no key-sig staff) if
    lilypond-book is unavailable, then to a zip of all images if
    pdflatex also fails.
    """
    song_title     = (form.get("song_title") or "").strip() or "Song Sheet"
    composer       = (form.get("composer")   or "").strip()
    key_tonic_raw  = (form.get("key_tonic")  or "").strip()
    key_mode       = (form.get("key_mode")   or "major").strip().lower()
    chord_list_raw = (form.get("chord_list") or "").strip()
    want_jazz      = form.get("guitar_jazz") == "on"
    want_pedal     = form.get("pedal_voicings") == "on"
    want_caged     = form.get("guitar_caged") == "on"
    want_piano     = form.get("piano") == "on"
    optional_fifth = form.get("optional_fifth") == "on"
    want_scale_kb  = form.get("scale_keyboard") == "on"
    want_scale_fret= form.get("scale_fretboard") == "on"
    want_chart     = form.get("lead_sheet") == "on"
    rootless_gtr   = form.get("rootless_guitar") == "on"
    want_subs      = form.get("chord_subs") == "on"
    rootless_kbd   = form.get("rootless_piano") == "on"
    transpose_to   = (form.get("transpose_to") or "").strip()

    if not chord_list_raw:
        return False, "Please enter at least one chord.", None, False

    # The chord list is now lead-sheet format: directives, bars of
    # changes, and chord definitions, in any order. A list of nothing but
    # definitions -- the old format -- parses to zero bars and renders
    # exactly as it always did.
    song = leadsheet.parse_leadsheet(chord_list_raw)

    # Directives fill in whatever the form left blank; an explicit form
    # field wins, since that is the one the user can see.
    if not (form.get("song_title") or "").strip() and song["title"]:
        song_title = song["title"]
    if not composer and song["composer"]:
        composer = song["composer"]
    if not key_tonic_raw and song["key"]:
        parsed_key = leadsheet.parse_key(song["key"])
        if parsed_key:
            key_tonic_raw = song["key"].split()[0]
            key_mode = parsed_key[1]

    # Transposition happens here, on the parsed song, because LilyPond's
    # \transpose only moves pitches: it shifts the key signature and
    # leaves a markup chord symbol reading whatever it read before.
    # Doing it now means the chart, the definitions and therefore the
    # diagrams all move together.
    if not transpose_to and song.get("transpose"):
        transpose_to = song["transpose"]
    if transpose_to:
        error = leadsheet.transpose_song(song, transpose_to)
        if error:
            return False, error, None, False
        parsed_key = leadsheet.parse_key(song["key"])
        if parsed_key:
            key_tonic_raw = song["key"].split()[0]
            key_mode = parsed_key[1]

    chart_score_block = None
    if want_chart:
        if not song["bars"]:
            return False, (
                "No bars found. A lead sheet needs at least one line of "
                "changes, e.g.  | Cm7 | F7 | Bbmaj7 |"
            ), None, False
        if cd.lilypond_book_path() is None:
            # No graceful degradation here: the chart IS LilyPond. Say so
            # rather than quietly returning a sheet without it.
            return False, (
                "The lead-sheet chart needs lilypond-book, which was not "
                "found on this server. Untick 'Lead sheet chart' to "
                "generate the diagrams on their own."
            ), None, False

    # Parse the key tonic if provided.
    key_score_block = None
    key_tonic_letter = key_tonic_accidental = None
    if key_tonic_raw:
        try:
            tonic_parsed = cd.parse_note(key_tonic_raw.split()[0])
            key_tonic_letter     = tonic_parsed[0]
            key_tonic_accidental = tonic_parsed[1]
            key_score_block = cd.key_signature_score_for_key(
                key_tonic_letter, key_tonic_accidental, key_mode
            )
        except (ValueError, IndexError):
            pass

    # Key scale diagrams (keyboard and/or guitar fretboard)
    key_scale_images = []
    if key_tonic_letter and (want_scale_kb or want_scale_fret):
        slug_prefix = cd.slugify(f"{key_tonic_raw} {key_mode}")
        key_scale_images = cd.draw_key_scale_diagrams(
            key_tonic_letter, key_tonic_accidental, key_mode,
            slug_prefix,
            want_keyboard=want_scale_kb,
            want_fretboard=want_scale_fret,
        )

    # Parse "Name: spelling" lines; skip blanks and comment lines (#).
    chord_entries = []
    errors = []
    if want_chart:
        chart_score_block = leadsheet.build_leadsheet_score(song)

    chord_entries, errors, chord_warnings = _parse_chord_list_from_song(
        song, want_subs=want_subs)
    song["warnings"].extend(chord_warnings)

    # Advisory notes, collected after the chords are resolved so an
    # unspellable symbol is reported alongside any bar that did not add
    # up. A sheet with either still builds; the user is just told.
    warning_note = ""
    if song["warnings"]:
        warning_note = "\n\nNotes on the changes:\n" + "\n".join(
            f"  - {w}" for w in song["warnings"]
        )

    if errors:
        return False, "Chord list errors:\n" + "\n".join(errors), None, False
    if not chord_entries and not song["bars"]:
        return False, "No valid chords found.", None, False

    # Generate diagrams for each chord.
    all_images = list(key_scale_images)   # flat list for zip fallback; scale imgs first
    for entry in chord_entries:
        name        = entry["name"]
        parsed      = entry["parsed"]
        inversions  = entry["inversions"]
        slug        = cd.slugify(name)

        entry["guitar_images"]   = []
        entry["keyboard_images"] = []

        # Guitar: jazz shapes
        if want_jazz:
            jazz_imgs = cd.draw_jazz_shapes(
                parsed, name, slug,
                optional_fifth=optional_fifth,
                max_voicings=1,
                include_open_pedal=want_pedal,
                rootless=rootless_gtr,
            )
            entry["guitar_images"].extend(jazz_imgs)
            all_images.extend(jazz_imgs)

        # Guitar: CAGED (triads only)
        if want_caged:
            root_letter     = parsed[0][0]
            root_accidental = parsed[0][1]
            root_pc         = (cd.PITCH_CLASS[root_letter] + root_accidental) % 12
            third_pc = (cd.PITCH_CLASS[parsed[1][0]] + parsed[1][1]) % 12 \
                if len(parsed) > 1 else (root_pc + 4) % 12
            fifth_pc = (cd.PITCH_CLASS[parsed[2][0]] + parsed[2][1]) % 12 \
                if len(parsed) > 2 else (root_pc + 7) % 12
            triad_q = cd.classify_triad(root_pc, third_pc, fifth_pc)[1]
            if triad_q in ('maj', 'min'):
                caged_imgs = cd.draw_all_caged_shapes(
                    root_letter, root_accidental, triad_q, name, slug
                )
                entry["guitar_images"].extend(caged_imgs)
                all_images.extend(caged_imgs)

        # Piano: keyboard diagram(s) for each requested inversion
        if want_piano:
            kb_imgs = cd.draw_keyboard_inversions(
                parsed, name, slug, inversions, rootless=rootless_kbd)
            entry["keyboard_images"].extend(kb_imgs)
            all_images.extend(kb_imgs)

    output_stem = cd.slugify(song_title)

    # --- Path 1: lilypond-book + pdflatex (full, with key sig staff) ---
    import shutil as _shutil
    if _shutil.which("lilypond-book") or os.path.isfile("/usr/bin/lilypond-book"):
        lytex_filename = f"{output_stem}-song-sheet.lytex"
        tex_source = cd.build_song_sheet_lytex(
            song_title, composer, key_score_block, chord_entries,
            key_scale_images=key_scale_images,
            chart_score_block=chart_score_block,
        )
        with open(lytex_filename, "w") as f:
            f.write(tex_source)

        out_dir = f"{output_stem}-lb-out"
        success, message = cd.try_run_lilypond_book(
            lytex_filename, all_images, out_dir=out_dir
        )
        if success:
            tex_stem = lytex_filename.replace(".lytex", "")
            pdf_path = os.path.join(work_dir, out_dir,
                                    f"{tex_stem}.pdf")
            return True, message + warning_note, pdf_path, False

    # --- Path 2: plain pdflatex (no key sig staff) ---
    tex_filename = f"{output_stem}-song-sheet.tex"
    tex_source = cd.build_song_sheet_tex(song_title, chord_entries,
                                         key_scale_images=key_scale_images)
    with open(tex_filename, "w") as f:
        f.write(tex_source)

    success, message, pdf_path = cd.try_run_pdflatex(tex_filename)
    if success:
        pdf_path = os.path.join(work_dir, pdf_path)
        return True, message + warning_note, pdf_path, False

    # --- Path 3: zip fallback ---
    zip_base = os.path.join(
        tempfile.gettempdir(),
        f"{output_stem}-{uuid.uuid4().hex[:8]}"
    )
    zip_path = shutil.make_archive(zip_base, "zip", work_dir)
    return False, message, zip_path, True


if __name__ == "__main__":
    # debug=False: the Werkzeug debugger exposes an interactive Python
    # console in the browser on any unhandled error, protected only by
    # a PIN shown in the terminal -- fine for active development, not
    # something to leave on for a finished tool, even one meant to run
    # locally only. Flip to True if you're modifying this file and want
    # auto-reload + better tracebacks.
    app.run(debug=False, port=5000)
