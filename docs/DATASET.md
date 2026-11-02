# Uzbek Sign Language dataset & training

> **This repository contains no real Uzbek Sign Language recordings.** The model shipped for demos is trained on *synthetic* landmark data (`dataset/_synthetic`) and is flagged `synthetic: true` everywhere in the UI. It proves that data → training → ONNX → live recognition → UI works. It does **not** understand sign language. A real dataset must be recorded with Deaf signers (ideally with the Uzbek Society of the Deaf and certified interpreters).

## Layout
```
dataset/
  signs.json                    vocabulary registry (id, name, uz_latin, uz_cyrillic, category, pos)
  pasport/ali_001.mp4           optional source video   (folder = sign name lower-case)
  pasport/ali_001.npz           landmark sequence 'sequence' (T x 329)   <- what training reads
  salom/...
  _synthetic/                   generated test data (ignored when real data exists)
```
`<speaker>_<n>` in the file name gives the speaker id; with ≥ 4 speakers the train/val/test split is **speaker-independent** (honest accuracy). With fewer speakers the split is random and the trainer says the accuracy is optimistic. Advanced metadata (sign_id, sign_name, sequence, language, speaker_id, video_path, translation) is stored in the `sign_datasets` table for every sample recorded in the Admin panel.

## Collecting samples
1. **Admin → Signs & dataset → Record**: pick a desk with an online AI service, enter a speaker id, press *Record*. After a 2-second countdown the AI service records landmarks for the chosen duration and uploads them (no raw video is stored by this path).
2. Or put videos into `dataset/<sign>/<speaker>_<n>.mp4` and run `python -m training.extract_landmarks`.
3. Aim for ≥ 30 samples per sign from ≥ 8 different signers, varied lighting, clothing, distance, left- and right-handed.

## Adding a new sign (no code change)
Admin → *Add a new sign class* (or append to `dataset/signs.json`), record samples, then **Training → Start training**. For new *sentence patterns* add a template to `ai/nlp.py: TEMPLATES`.

## Training
```
python -m training.train --epochs 80 --arch transformer     # transformer | gru | tcn
python -m training.train --allow-synthetic --epochs 25      # pipeline test only
```
Outputs `models/<version>/` with `model.onnx`, `model.pt`, `meta.json` (classes, split, metrics, per-class precision/recall, confusion matrix, training args) and `confusion_matrix.png`. Confidence is **temperature-calibrated on the validation set** so "94 %" means roughly 94 % correct. A model is auto-activated only if validation accuracy ≥ 0.85 (`--min-val-acc`); otherwise activate manually after review. Signs with < 5 samples are skipped and reported. Retrain whenever data grows; old versions stay available for rollback.

## Quality bar before real use
Evaluate on **unseen signers**, in the real room and camera position, with the confidence threshold chosen from the precision/recall table. Keep the employee in the loop (the system always shows the recognised glosses and confidence to the employee and never auto-acts on them).
