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
import pickle
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
            "start_sec": time_to_seconds(start),
            "end_sec": time_to_seconds(end),
            "start_time": start,
            "end_time": end,
            "text": text.strip('"')
        })

    return clips


# ======================================================
# 4. Load all subtitle files
# ======================================================
def load_all_subtitles():
    SUBTITLE_DIR = "Processed Movie Subtitles"
    raw_clips = []

    for file in os.listdir(SUBTITLE_DIR):
        if file.endswith(".srt"):
            path = os.path.join(SUBTITLE_DIR, file)
            raw_clips.extend(parse_srt(path))

    print(f"Raw subtitle clips loaded: {len(raw_clips)}")
    return raw_clips


# ======================================================
# 5. Build CONTEXTUAL clips
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
            "start_time": clips[start_idx]["start_time"],
            "end_time": clips[end_idx - 1]["end_time"],
            "start_sec": clips[start_idx]["start_sec"],
            "end_sec": clips[end_idx - 1]["end_sec"],
            "text": merged_text
        })

    return contextual


# ======================================================
# 6. Behavioral bonus
# ======================================================
def behavior_bonus(text):
    bonus = 0.0
    t = text.strip()
    tl = t.lower()
    words = t.split()

    if "..." in t:
        bonus += 0.06
    if t.endswith(("…", ".", ",")):
        bonus += 0.03
    if any(tl.startswith(w) for w in ["uh", "um", "well", "so ", "okay", "hmm"]):
        bonus += 0.05
    if t.count("?") >= 2:
        bonus += 0.05
    if any(p in tl for p in ["what do you mean", "are you saying", "how is that", "why would"]):
        bonus += 0.04
    if "..." in t and "?" in t:
        bonus += 0.04
    if len(words) <= 4 and "?" not in t:
        bonus += 0.04
    if any(c in t for c in ["—", "--"]):
        bonus += 0.04
    if "," in t and len(words) > 6:
        bonus += 0.02
    if t.endswith("?") and not tl.startswith(("why", "what", "how")):
        bonus += 0.03
    if any(tl.startswith(w) for w in ["i think", "maybe", "it depends", "sort of"]):
        bonus += 0.05
    if "!" in t and "?" in t:
        bonus += 0.04
    if t.isupper() and len(words) > 2:
        bonus += 0.03
    if any(w in tl for w in ["wait", "hold on", "let me think", "give me a second"]):
        bonus += 0.05
    if t.count(",") >= 2:
        bonus += 0.03
    if t.startswith(("...", "—", "-")):
        bonus += 0.05
    if len(words) <= 3 and t.endswith("."):
        bonus += 0.04
    if t.startswith(("but", "and", "no,")):
        bonus += 0.03
    if "/" in t or t.count("-") >= 2:
        bonus += 0.03
    if len(words) <= 5 and "..." in t:
        bonus += 0.05
    if len(words) >= 12 and t.count(",") >= 2:
        bonus += 0.04

    return min(bonus, 0.25)


# ======================================================
# 7. Confidence conversion
# ======================================================
def cosine_to_confidence(score):
    return max(0.0, min(float(score), 1.0)) * 100


# ======================================================
# 8. Initialize model and index (GLOBAL)
# ======================================================
# ======================================================
# 8. Initialize model and index (GLOBAL)
# ======================================================
CACHE_DIR = "cache"
CLIPS_FILE = os.path.join(CACHE_DIR, "clips.pkl")
INDEX_FILE = os.path.join(CACHE_DIR, "behavenet.index")

def initialize_system():
    global model, index, clips
    
    # 1. Load Model (Fast enough to load every time, ~2-5s)
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # 2. Check Cache
    if os.path.exists(CLIPS_FILE) and os.path.exists(INDEX_FILE):
        print("Found cached data. Loading...")
        try:
            with open(CLIPS_FILE, "rb") as f:
                clips = pickle.load(f)
            
            index = faiss.read_index(INDEX_FILE)
            print(f"✅ Loaded {len(clips)} clips and index with {index.ntotal} vectors from cache.")
            return
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
    
    # 3. Generate if no cache
    print("No cache found. Generating embeddings (this may take a while)...")
    
    # Ensure cache dir exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    raw_clips = load_all_subtitles()
    clips = build_contextual_clips(raw_clips, window=1)
    print(f"Contextual clips created: {len(clips)}")
    
    # Generate embeddings
    texts = [clip["text"] for clip in clips]
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    
    # Save to cache
    print("Saving data to cache...")
    with open(CLIPS_FILE, "wb") as f:
        pickle.dump(clips, f)
    
    faiss.write_index(index, INDEX_FILE)
    print("✅ System initialized and cached!")

# Run initialization
initialize_system()


# ======================================================
# 9. Search function (EXPORTABLE)
# ======================================================
def search(query, top_k=50, min_confidence=40):
    """
    Semantic search function
    
    Args:
        query: Search query string
        top_k: Number of top results to consider
        min_confidence: Minimum confidence threshold (0-100)
    
    Returns:
        List of results sorted by confidence
    """
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
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "start_sec": int(clip["start_sec"]),
            "confidence": round(confidence, 2),
            "text": clip["text"]
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return results


# ======================================================
# 10. Get dataset info (EXPORTABLE)
# ======================================================
def get_dataset_info():
    """Get dataset statistics"""
    return {
        "total_clips": len(clips),
        "unique_movies": len(set(c['movie'] for c in clips)),
        "model_name": "all-mpnet-base-v2",
        "embedding_dim": index.d
    }


# ======================================================
# 11. Interactive CLI (for testing)
# ======================================================
if __name__ == "__main__":
    print("\n🎬 Semantic Footage Search Engine (CLI Mode)")
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

        top_5 = results[:5]

        for i, r in enumerate(top_5, 1):
            print(f"\n{i}. 🎬 MOVIE       : {r['movie']}")
            print(f"   ⏱️  TIMESTAMPS  : {r['start_time']} → {r['end_time']}")
            print(f"   📊 CONFIDENCE  : {r['confidence']}%")
            print(f"   💬 DIALOGUE    : {r['text']}")

        print("\n")