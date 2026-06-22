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
        want_full_fretboard = form.get("full_fretboard") == "on"

        images, blocks = cd.render_chord_with_inversions(
            parsed_root, name, slug,
            how_many=how_many, want_full_chord=want_full_chord,
            want_info_panel=want_info_panel, want_guitar_shapes=want_guitar_shapes,
            want_full_fretboard=want_full_fretboard,
        )
        sub_entries.append((name, blocks, images, []))

    else:  # scale
        spelling = form.get("scale_spelling", "")
        tokens = spelling.split()
        if not tokens:
            return False, "Please enter a scale spelling.", None, False
        parsed = [cd.parse_note(t) for t in tokens]

        notes_with_octaves = cd.assign_octaves(parsed, start_octave=4)
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
                full_scale_voicing, root_letter, root_accidental
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


if __name__ == "__main__":
    # debug=False: the Werkzeug debugger exposes an interactive Python
    # console in the browser on any unhandled error, protected only by
    # a PIN shown in the terminal -- fine for active development, not
    # something to leave on for a finished tool, even one meant to run
    # locally only. Flip to True if you're modifying this file and want
    # auto-reload + better tracebacks.
    app.run(debug=False, port=5000)
