# Plan: finding the best photos in a Google Photos library

## Context

The library holds about 200,000 photos (600 GB) in Google Photos. The goal is to find a manageable shortlist of candidates
for printing or other special treatment, starting with a Google Takeout of the last ~5 years. No one can review 100k photos
by hand, so the approach is a funnel. Cheap automatic filters remove the obvious rejects. Learned quality and aesthetic
scores rank what is left. Event clustering keeps the shortlist varied. Claude reviews a few thousand finalists, and you
make the final call in a fast local review page. Everything runs locally on this desktop (SmallPinkDot: 16 cores, 47 GB
RAM, GTX 1650 4 GB, ffmpeg present, exiftool **not** installed). The repo `~/photo-archive` is currently empty apart from
a `.venv`, and it is not yet a git repo.

Why Takeout and not an API: since March 2025 the Google Photos Library API only exposes photos an app uploaded itself,
so Takeout is the only practical bulk route.

## Funnel (rough numbers for ~100k photos over 5 years)

| Stage | What it does | Left after it |
| --- | --- | --- |
| 0. Ingest | Unzip, index metadata into SQLite | ~100k |
| 1. Hard filters | Drop videos, screenshots, documents, low-res, blank images | ~70k |
| 2. Duplicates/bursts | Group near-duplicates, keep the best of each group | ~35k |
| 3. Scoring | Sharpness, exposure, technical-quality model, aesthetic model, faces | ranked |
| 4. Diversity | Cluster into events (time + place), take top N per event | ~3–5k |
| 5. Claude review | Claude looks at contact sheets of finalists, grades them and says why | ~500–1,000 |
| 6. Your review | Local gallery page with keyboard keep/reject, then export originals | your picks |

## Step by step

### 0. Get the data (your part, with guidance)
- In Takeout, select only Google Photos, then "All photo albums included" → pick `Photos from 2021` … `Photos from 2025`
  (plus any named albums you care about, since album membership is a useful signal). Choose .zip and **50 GB** parts.
- **Download to F: (2.3 TB free), not C: (only 14 GB free).** The browser defaults to `C:\Users\…\Downloads`, so change
  that first. Plan for ~2× the Takeout size (zips + extracted); you can delete the zips after a verified extraction.
- Links expire after 7 days, so download all the parts promptly.
- Suggested layout: `F:\PhotoArchive\takeout_zips\`, `F:\PhotoArchive\extracted\`, `F:\PhotoArchive\work\`
  (thumbnails, embeddings, SQLite DB). Only code and small config go in the git repo, never photos.

### 1. Repo setup (on approval)
- `git init`, add remote `https://github.com/KCaldeira/photo-archive.git`, `.gitignore` (data, thumbnails, DB, `.venv`,
  `.vscode/*` except `settings.json`), a `.vscode/settings.json` with a dark tint no other repo uses, and a short
  `CLAUDE.md` recording the data paths on F:.
- Python deps in `.venv`: `pillow`, `pillow-heif` (iPhone HEIC), `numpy`, `opencv-python-headless`, `imagehash`,
  `torch` (CUDA), `open_clip_torch`, `pyiqa`, `insightface` or `mediapipe` (faces), `scikit-learn`, `pandas`.
- Ask you to `sudo apt install libimage-exiftool-perl` (exiftool is not installed).

### 2. Ingest → SQLite (`ingest.py`)
- Walk the extracted tree. For each image record the path, size, dimensions, SHA-1, EXIF (date, camera, lens, ISO,
  exposure, GPS), and the Takeout JSON sidecar fields (`photoTakenTime`, `geoData`, `people` = named faces,
  `favorited` if present, album folder, whether a `-edited` version exists).
- Handle the known Takeout quirks: sidecar names truncated at ~46 chars, `(1)` suffix mismatches, Live/Motion photo
  `.MP4` companions, and the same photo appearing in both a year folder and an album folder (dedupe by hash).
- Make a 512 px thumbnail for every image (PIL JPEG draft mode makes this fast), using 14 worker processes under
  `nice -n 19`.

### 3. Hard filters (`filter.py`)
Mark photos as excluded, never delete them. Rules: videos; under ~6 MP (too small for a good 8×10); screenshots
(no camera EXIF plus screen-like dimensions or PNG); documents, receipts and whiteboards (detected with a CLIP zero-shot
prompt); near-black or near-white frames. Thresholds go in a config file and are tuned on the pilot.

### 4. Near-duplicates and bursts (`dedupe.py`)
- Compute CLIP ViT-L/14 image embeddings on the GPU (fp16 fits in 4 GB). They are reused for scoring, clustering and search.
- Group photos taken within a few seconds of each other whose embeddings have cosine similarity above ~0.95. Keep the
  best-scoring photo in each group and mark the rest as alternates, which stay visible in the review page.

### 5. Scoring (`score.py`)
Each photo gets these columns, then a weighted composite:
- **Technical**: sharpness (variance of the Laplacian, measured on the subject or face region where possible), exposure
  and clipping from the histogram, noise, and a no-reference IQA model from `pyiqa` (e.g. MUSIQ or TOPIQ).
- **Aesthetic**: the LAION aesthetic predictor (a small MLP on the same CLIP embeddings, so it costs almost nothing).
- **People**: face count, face size, face sharpness, eyes open/smile where the model supports it, and named people
  from Takeout `people` tags.
- **Your own signals**: favorited, in a named album, has an edited version. These are strong priors, so weight them up.
- **Print-readiness**: largest print size at 300 dpi and 200 dpi, computed from pixel dimensions.

### 6. Events and diversity (`events.py`)
Cluster photos into events by time gaps (e.g. more than 3 h apart starts a new event) refined by GPS distance. Choose
finalists as the top-k per event, with k scaled by the event's size and best score, plus a global top-N. This stops
one beach trip from filling the whole shortlist.

### 7. Claude review of finalists
- Build contact sheets of about 16–20 numbered thumbnails per sheet. I (Claude Code, in a session) can read the sheets
  directly and grade each photo A/B/C with a one-line reason (moment, composition, light, emotional value, works as a
  print?). Grades are written back to SQLite.
- 3–5k finalists means roughly 200–300 sheets. That fits a few sessions, or it can be batched through the Claude API
  if you prefer it unattended.
- Only small thumbnails are viewed. Nothing is uploaded anywhere else.

### 8. Your review page (`review.py`) and export
- A local static HTML gallery generated from SQLite, served with `python -m http.server`. Photos are private, so it
  stays local and is not published as an artifact.
- Keys: keep / reject / star 1–5 / show burst alternates / open the full-resolution image. Filters by year, event,
  person, score and Claude grade.
- Free-text search over CLIP embeddings ("kids on the beach at sunset", "snow mountains") to build themed sets for
  books or calendars.
- Decisions are saved to a JSON file that `import_decisions.py` loads into SQLite. `export.py` copies the chosen
  originals (the edited version where one exists) to `F:\PhotoArchive\selects\<set>\`, with a CSV listing max print sizes.

## Pilot first
Run the whole pipeline on **one year (or one 50 GB zip)** before doing all 5 years. Check a few hundred photos by eye
against the filters and scores, tune thresholds and weights, check how well Claude's grades match your own picks, and
only then scale up. The pipeline is incremental: everything is keyed on file hash, so adding more years later
(including the older 15 years) only processes new files.

## Time and resource estimates (desktop)
- Thumbnails and EXIF for 100k photos: ~20–40 min on 14 cores, niced.
- CLIP embeddings plus IQA on the GTX 1650: ~2–4 h for 100k. Run it in the background with `nice -n 19` and
  `OMP_NUM_THREADS=1` per worker.
- Disk: the Takeout size × 2, plus ~10 GB for thumbnails and the work DB.

## Verification
- After ingest: row count equals the image file count, sidecar match rate above 98%, dates sane (no 1970 or future dates).
- Filters: a random sample of 200 excluded photos shown on a contact sheet, so you can confirm almost no keepers
  were dropped.
- Scoring: a sheet of the top 50 and bottom 50 shows clearly that the ranking works. Check the composite score against
  your Google favorites (they should rank high).
- End to end on the pilot year: the review page loads, decisions persist, and the export copies the right originals.

## Open choices (defaults in brackets, adjustable after the pilot)
- What "best" means to you: people/family vs travel/landscape vs both [both, with people weighted up].
- Claude review in-session vs via the API [in-session for the pilot].
- Data location [F:\PhotoArchive].
