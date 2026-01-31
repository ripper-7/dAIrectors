# ======================================================
# 0. CRASH FIXES (MUST BE AT VERY TOP)
# ======================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

# ======================================================
# 1. Imports
# ======================================================
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ======================================================
# 2. Timestamp conversion
# ======================================================
def time_to_seconds(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ======================================================
# 3. Parse SRT file (robust)
# ======================================================
def parse_srt(file_path):
    clips = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_range = lines[1].strip()
        if "-->" not in time_range:
            continue

        start, end = [t.strip() for t in time_range.split("-->")]

        text = " ".join(
            line.strip() for line in lines[2:]
            if line.strip() and not line.strip().isdigit()
        )

        if not text:
            continue

        clips.append({
            "movie": os.path.basename(file_path),
        
            # numeric (for internal use if ever needed)
            "start_sec": time_to_seconds(start),
            "end_sec": time_to_seconds(end),
        
            # original timestamps (for display)
            "start_time": start,
            "end_time": end,
        
            "text": text.strip('"')
        })


    return clips


# ======================================================
# 4. Load all subtitle files
# ======================================================
SUBTITLE_DIR = "Movie Subtitles"

raw_clips = []

for file in os.listdir(SUBTITLE_DIR):
    if file.endswith(".srt"):
        path = os.path.join(SUBTITLE_DIR, file)
        raw_clips.extend(parse_srt(path))

print(f"Raw subtitle clips loaded: {len(raw_clips)}")


# ======================================================
# 5. Build CONTEXTUAL clips (KEY FIX)
# ======================================================
def build_contextual_clips(clips, window=1):
    contextual = []

    for i in range(len(clips)):
        start_idx = max(0, i - window)
        end_idx = min(len(clips), i + window + 1)

        merged_text = " ".join(
            clips[j]["text"] for j in range(start_idx, end_idx)
        )

        contextual.append({
            "movie": clips[i]["movie"],

            # keep original SRT timestamps
            "start_time": clips[start_idx]["start_time"],
            "end_time": clips[end_idx - 1]["end_time"],

            # numeric (optional, not printed)
            "start_sec": clips[start_idx]["start_sec"],
            "end_sec": clips[end_idx - 1]["end_sec"],

            "text": merged_text
        })

    return contextual



clips = build_contextual_clips(raw_clips, window=1)
print(f"Contextual clips created: {len(clips)}")


# ======================================================
# 6. Load embedding model
# ======================================================
model = SentenceTransformer("all-mpnet-base-v2")


# ======================================================
# 7. Generate embeddings
# ======================================================
texts = [clip["text"] for clip in clips]

embeddings = model.encode(
    texts,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


# ======================================================
# 8. Build FAISS index (cosine similarity)
# ======================================================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

print("Vectors indexed:", index.ntotal)


# ======================================================
# 9. Confidence conversion
# ======================================================
def cosine_to_confidence(score):
    return max(0.0, min(float(score), 1.0)) * 100


# ======================================================
# 10. Behavioral bonus (THIRD FIX)
def behavior_bonus(text):
    bonus = 0.0
    t = text.strip()
    tl = t.lower()
    words = t.split()

    # --------------------------------------------------
    # 1️⃣ Hesitation / Delay / Uncertainty
    # --------------------------------------------------
    if "..." in t:
        bonus += 0.06  # trailing off / pause

    if t.endswith(("…", ".", ",")):
        bonus += 0.03  # unfinished thought

    if any(tl.startswith(w) for w in ["uh", "um", "well", "so ", "okay", "hmm"]):
        bonus += 0.05  # hesitation starters

    # --------------------------------------------------
    # 2️⃣ Confusion / Misunderstanding
    # --------------------------------------------------
    if t.count("?") >= 2:
        bonus += 0.05  # repeated questioning = confusion

    if any(p in tl for p in ["what do you mean", "are you saying", "how is that", "why would"]):
        bonus += 0.04  # clarification-seeking structure

    # --------------------------------------------------
    # 3️⃣ Awkwardness / Social Tension
    # --------------------------------------------------
    if "..." in t and "?" in t:
        bonus += 0.04  # question + pause = awkwardness

    if len(words) <= 4 and "?" not in t:
        bonus += 0.04  # short, flat reply

    # --------------------------------------------------
    # 4️⃣ Emotional Shift / Subtle Change
    # --------------------------------------------------
    if any(c in t for c in ["—", "--"]):
        bonus += 0.04  # interruption / tonal shift

    if "," in t and len(words) > 6:
        bonus += 0.02  # mid-sentence pivot

    # --------------------------------------------------
    # 5️⃣ Defensive / Avoidant Responses
    # --------------------------------------------------
    if t.endswith("?") and not tl.startswith(("why", "what", "how")):
        bonus += 0.03  # deflective question

    if any(tl.startswith(w) for w in ["i think", "maybe", "it depends", "sort of"]):
        bonus += 0.05  # vagueness

    # --------------------------------------------------
    # 6️⃣ Conflict / Tension Build-Up
    # --------------------------------------------------
    if "!" in t and "?" in t:
        bonus += 0.04  # emotional instability

    if t.isupper() and len(words) > 2:
        bonus += 0.03  # raised voice / emphasis

    # --------------------------------------------------
    # 7️⃣ Real-Time Thinking / Processing
    # --------------------------------------------------
    if any(w in tl for w in ["wait", "hold on", "let me think", "give me a second"]):
        bonus += 0.05

    if t.count(",") >= 2:
        bonus += 0.03  # self-correction / thinking aloud

    # --------------------------------------------------
    # 8️⃣ Reaction-First Moments
    # --------------------------------------------------
    if t.startswith(("...", "—", "-")):
        bonus += 0.05  # silent reaction before speech

    if len(words) <= 3 and t.endswith("."):
        bonus += 0.04  # delayed, minimal response

    # --------------------------------------------------
    # 9️⃣ Conversational Dynamics
    # --------------------------------------------------
    if t.startswith(("but", "and", "no,")):
        bonus += 0.03  # interruption / overlap

    if "/" in t or t.count("-") >= 2:
        bonus += 0.03  # overlapping dialogue artifacts

    # --------------------------------------------------
    # 🔟 Meta / Editor-Friendly Moments
    # --------------------------------------------------
    if len(words) <= 5 and "..." in t:
        bonus += 0.05  # quiet beat

    if len(words) >= 12 and t.count(",") >= 2:
        bonus += 0.04  # emotional buildup before reply

    # Safety clamp (important)
    return min(bonus, 0.25)



# ======================================================
# 11. Search function (semantic + behavioral)
# ======================================================
def search(query, top_k=50, min_confidence=40):
    query_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_emb, top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        raw_score = float(scores[0][i])
        clip = clips[idx]

        final_score = raw_score + behavior_bonus(clip["text"])
        confidence = cosine_to_confidence(final_score)

        if confidence < min_confidence:
            continue

        results.append({
            "movie": clip["movie"],
            "start_time": clip["start_time"],   # SRT format
            "end_time": clip["end_time"],       # SRT format
            "confidence": round(confidence, 2),
            "text": clip["text"]
        })


    # 🔑 IMPORTANT: sort by confidence
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return results


# ======================================================
# 12. Interactive semantic search (USER INPUT)
# ======================================================
print("\n🎬 Semantic Footage Search Engine")
print("Type a natural language query (or 'exit' to quit)\n")

while True:
    user_query = input("🔎 Enter your query: ").strip()

    if user_query.lower() in ["exit", "quit"]:
        print("\nExiting search. Goodbye 👋")
        break

    print("\n" + "=" * 80)
    print(f"🔎 QUERY: {user_query.upper()}")
    print("=" * 80)

    results = search(user_query)

    if not results:
        print("No strong matches found.\n")
        continue

    top_5 = results[:5]   # ✅ TOP 5 BY CONFIDENCE

    for i, r in enumerate(top_5, 1):
        print(f"\n{i}. 🎬 MOVIE       : {r['movie']}")
        print(f"   ⏱️  TIMESTAMPS  : {r['start_time']} → {r['end_time']}")
        print(f"   📊 CONFIDENCE  : {r['confidence']}%")
        print(f"   💬 DIALOGUE    : {r['text']}")

    print("\n")  # spacing for next query
