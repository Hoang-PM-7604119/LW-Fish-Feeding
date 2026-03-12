# Split Method: U-FFIA vs Reorganized Project

Comparison of the original [U-FFIA](https://github.com/FishMaster93/U-FFIA) split logic with our reorganized pipeline, and data leakage / bias risks.

---

## 1. U-FFIA (original) – video-only split

**Source:** `dataset/fish_video_dataset.py` in [FishMaster93/U-FFIA](https://github.com/FishMaster93/U-FFIA).

### How it works

- **Input:** Video paths only (no audio). Paths are collected per class via `get_video_name(split='strong'|'medium'|'weak'|'none')` with `glob` under `path/dir/dir1/<class>/*.mp4`.
- **Split:** For each class, the list of video paths is shuffled with a fixed seed, then split by **index**:
  - `test = list[:700]`
  - `val  = list[700:1400]`
  - `train = list[1400:]`
- **Unit of split:** One **video path** → appears in exactly one of train/val/test.

### Leakage / bias

| Risk | U-FFIA (video-only) |
|------|----------------------|
| Same video in train and test | **No** – each path is in one split only. |
| Reproducibility | **Unstable** – `os.listdir()` and `glob.glob()` order can vary by OS/FS, so same seed can give different splits. |
| Fixed splits | **No** – splits are recomputed every run; no saved split file. |
| Class balance | **Yes** – 700 test, 700 val, rest train per class. |
| Label order | none:0, **strong:1, medium:2, weak:3** (different from our convention below). |

So for **video-only**, the original U-FFIA design does **not** have path leakage; the main issue is reproducibility of the split across machines.

---

## 2. Reorganized project – multimodal (video + audio) split

We work with **(video, audio) pairs** and optional **fixed split file**.

### 2a. Dynamic split (no `split_file`)

- **Input:** Pairs built by matching video and audio with the same identifier (e.g. date/feed/name) **per class**.
- **Split:** Per class, pairs are shuffled then split by index (e.g. test 700, val 700, rest train), same idea as U-FFIA but applied to **pairs**.
- **Unit of split:** A **(video_path, audio_path)** pair.

**Leakage:**  
Because the same **video** or same **audio** can appear in **multiple pairs** (same id in different folders/sessions), one path can end up in both train and test when we split by pair index → **same file in train and test** → leakage for video-only or audio-only evaluation.

### 2b. Fixed split (saved `splits.json`) – pair-based creation

- **Input:** Same pairing logic; pairs saved in a JSON split file.
- **Creation (old):** For each class, shuffle pairs and assign by index to train/val/test (e.g. first 700 test, next 700 val, rest train).
- **Unit of split:** Again the **(video_path, audio_path)** pair.

**Leakage:**  
Same as 2a: the same video or audio path can appear in several pairs that are assigned to different splits → **path-level leakage** (e.g. 2778 audio paths in multiple splits), so video-only or audio-only test scores can be inflated.

### 2c. Fixed split – disjoint video and audio (recommended)

- **Idea:** Split so that each **video path** and each **audio path** appears in **at most one** of train/val/test.
- **Method:** Treat pairs as a graph (two pairs connected if they share a video or audio path). Assign **connected components** to splits (e.g. by class balance), so all pairs that share any path go to the same split.
- **Unit of split:** Effectively the **path** (each path in one split); pairs follow their paths.

**Leakage:**  
**No path leakage** – no video or audio file appears in more than one split, so video-only, audio-only, and fusion evaluation are all consistent and not inflated by same-file reuse.

---

## 3. Side-by-side summary

| Aspect | U-FFIA (original) | Reorganized (pair-based, old) | Reorganized (disjoint paths) |
|--------|--------------------|-------------------------------|------------------------------|
| Modality | Video only | Video + audio (pairs) | Video + audio (pairs) |
| Unit of split | Video path | (video, audio) pair | Path (via components) |
| Same path in 2 splits? | No | **Yes** (leakage) | No |
| Reproducible splits? | Unstable (glob/listdir) | Yes (fixed JSON) | Yes (fixed JSON) |
| Bias risk | Low (path-unique) | **High** (path reuse) | Low (path-unique) |

---

## 4. Label order difference

- **U-FFIA:** `none:0, strong:1, medium:2, weak:3`
- **Ours:** `none:0, weak:1, medium:2, strong:3`

If you compare metrics or load U-FFIA pretrained heads, the class indices do **not** match; remap labels or heads accordingly.

---

## 5. Recommendations

1. **Use a fixed split file** (as we do) so experiments are reproducible and comparable.
2. **Create splits with disjoint video and audio paths** (connected-component or equivalent logic) so there is no path-level leakage and video-only test results are not biased.
3. **Run `scripts/check_data_leakage.py`** on your `splits.json` to confirm:
   - No (video_file, audio_file) pair in more than one split.
   - No video or audio path in more than one split.
4. If you align with U-FFIA paper results, keep in mind their **video-only** split has no path leakage by construction; our **pair-based** split without disjoint paths does have leakage until you switch to disjoint-path creation.

---

## References

- U-FFIA repo: https://github.com/FishMaster93/U-FFIA  
- U-FFIA dataset (Zenodo): https://zenodo.org/records/11059975  
- Paper: https://arxiv.org/pdf/2309.05058.pdf  
