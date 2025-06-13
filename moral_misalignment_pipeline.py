#!/usr/bin/env python3
"""
Moral-Misalignment-Pipeline
────────────────────────────────────────────────────────────────────────
• API-Fetch  :  conversation_id-basiertes Paging (v2 recent search)
• Checkpoint :  JSON-Zwischenspeicher nach jedem Batch (Crash-Resistent)
• Filter     :  technische Filter  +  Moral Foundations Dictionary
• Ziel       :  exakt 150 valide Replies pro Event
• Analyse    :  BERTopic-Cluster  +  GPT-4o-Summaries
• Export     :  CSV / JSON / Modell-Ordner pro Event
────────────────────────────────────────────────────────────────────────
Autor:  Max Dorfmann        Stand: 13 Jun 2025
"""

from __future__ import annotations

# ───────── Basics ──────────────────────────────────────────────────
import argparse, json, logging, os, random, re, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd, tweepy, spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════
# 1  KONFIGURATION
# ══════════════════════════════════════════════════════════════════
BEARER_TOKEN = os.getenv("TW_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONFIG: Dict[str, Dict] = {
    "events": {
        "CE1": {
            "tweet_id": 1933177571598086553,
        },
        "CE2": {
            "tweet_id": 1931855149338906780,
        },
    },
    "target_replies": 150,
    "mfd_dic_path": "moral_foundations_dictionary.dic",
    "bertopic_model": "all-MiniLM-L6-v2",
    "random_seed": 137,
    "gpt_model": "gpt-4o",
    "gpt_temperature": 0.15,
    "output_dir": "results",
    "checkpoint_dir": ".checkpoints",
}

# ══════════════════════════════════════════════════════════════════
# 2  LOGGING
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO, # logging.DEBUG wenn alle erhaltene Replys geloggt werden sollen
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("moral_pipeline.log", "w", "utf-8"),
              logging.StreamHandler(sys.stdout)],
)

# ══════════════════════════════════════════════════════════════════
# 3  MFD – PARSING + CHECK
# ══════════════════════════════════════════════════════════════════
def parse_mfd(path: str | Path) -> Tuple[Set[str], List[str]]:
    full, pref = set(), []
    collecting = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ln = line.strip()
            if not ln:
                continue
            if ln.startswith("%"):
                if not collecting:
                    collecting = True  # Erste % => beginne beim nächsten
                    continue
                else:
                    break  # Zweite % => fertig mit Header, starte eigentlichen Bereich
        # Jetzt in der zweiten Schleife: Begriffe parsen
        for line in fh:
            ln = line.strip()
            if not ln or ln.startswith("%"):
                continue
            term = ln.split()[0].lower()
            if term.endswith("*"):
                pref.append(term[:-1])
            else:
                full.add(term)
    logging.info("MFD geladen: %d exakte, %d Präfixe", len(full), len(pref))
    return full, pref


def contains_mfd(text: str, full: Set[str], pref: List[str], nlp) -> bool:
    doc = nlp(text)
    for tok in doc:
        if not tok.is_alpha:
            continue
        lemma = tok.lemma_.lower()
        if lemma in full or any(lemma.startswith(p) for p in pref):
            return True
    return False

# ══════════════════════════════════════════════════════════════════
# 4  TECHNISCHE FILTER
# ══════════════════════════════════════════════════════════════════
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")

def clean(txt: str) -> str:
    return MENTION_RE.sub("", URL_RE.sub("", txt)).strip()

def tech_ok(txt: str) -> bool:
    if len(URL_RE.findall(txt)) > 1:
        return False
    if len(clean(txt).split()) < 5:
        return False
    return True

# ══════════════════════════════════════════════════════════════════
# 5  CHECKPOINT-HELPER
# ══════════════════════════════════════════════════════════════════
def cp_dir(event_key: str) -> Path:
    d = Path(CONFIG["checkpoint_dir"]) / event_key
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_checkpoint(event_key: str) -> Tuple[List[str], str | None]:
    p = cp_dir(event_key) / "accepted.json"
    token_p = cp_dir(event_key) / "next_token.txt"
    replies = json.loads(p.read_text()) if p.exists() else []
    next_token = token_p.read_text().strip() if token_p.exists() else None
    return replies, next_token

def save_checkpoint(event_key: str, replies: List[str], next_token: str | None):
    (cp_dir(event_key) / "accepted.json").write_text(json.dumps(replies, ensure_ascii=False, indent=2))
    if next_token:
        (cp_dir(event_key) / "next_token.txt").write_text(next_token)
    else:
        (cp_dir(event_key) / "next_token.txt").unlink(missing_ok=True)

# ══════════════════════════════════════════════════════════════════
# 6  TWITTER API FETCH
# ══════════════════════════════════════════════════════════════════
def fetch_replies_api(event_key: str, tweet_id: int, target: int, nlp, full, pref) -> List[str]:
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

    accepted, next_token = load_checkpoint(event_key)
    seen = set(accepted)
    logging.info("▶ Checkpoint geladen – %d valid Replies", len(accepted))

    query = f"in_reply_to_tweet_id:{tweet_id} lang:en -is:retweet"
    fields = ["author_id", "created_at", "in_reply_to_user_id", "public_metrics"]
    logging.info("API-Query: %s", query)

    while len(accepted) < target:
        try:
            resp = client.search_recent_tweets(
                query=query,
                tweet_fields=fields,
                max_results=100,
                next_token=next_token,
            )
        except tweepy.TooManyRequests:
            logging.warning("Rate-Limit hit – warte 60 s")
            time.sleep(60)
            continue

        if resp.data is None:
            logging.warning("Keine weiteren Tweets von API.")
            break

        for tw in resp.data:
            txt = tw.text.strip()
            if txt in seen:
                continue
            seen.add(txt)
            logging.debug(f"Reply: {txt}")
            if tech_ok(txt) and contains_mfd(txt, full, pref, nlp):
                accepted.append(txt)
                if len(accepted) >= target:
                    break

        next_token = resp.meta.get("next_token")
        logging.info("… Stand: %d valid Replies | next_token=%s",
                     len(accepted), next_token or "␀")

        save_checkpoint(event_key, accepted, next_token)
        if not next_token:
            break

    logging.info("✓ Final valid Replies: %d", len(accepted))
    return accepted[:target]

# ══════════════════════════════════════════════════════════════════
# 7  GPT-4o TOPIC-SUMMARY
# ══════════════════════════════════════════════════════════════════
def gpt_summary(docs: List[str]) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = ("Interpretiere den folgenden thematisch kohärenten Cluster an repräsentativen Tweets.\n\nTweets:\n"
              + "\n---\n".join(docs[:10]))
    rsp = client.chat.completions.create(
        model=CONFIG["gpt_model"],
        temperature=CONFIG["gpt_temperature"],
        messages=[{"role": "system", "content": "Du bist Analyse-Assistent und es geht um die Interpretation zu Moral-Missalignment trotz Continued Usage im Kontext des Brand Leaders Elon Musk und seinen Tweets auf der Plattform x.com. Die Interpretation soll eine Länge von 2-3 Sätzen nicht überschreiten und theorie-neutral gehalten werden."},
                  {"role": "user",   "content": prompt}],
    )
    return rsp.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════
# 8  EVENT-PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_event(event_key: str, nlp, full, pref):
    cfg = CONFIG["events"][event_key]
    logging.info("Event %s | Tweet %s", event_key, cfg["tweet_id"])

    replies = fetch_replies_api(
        event_key=event_key,
        tweet_id=cfg["tweet_id"],
        target=CONFIG["target_replies"],
        nlp=nlp, full=full, pref=pref,
    )

    if len(replies) < CONFIG["target_replies"]:
        logging.warning("%d/%d Replies – Event übersprungen", len(replies), CONFIG["target_replies"])
        return

    # ─── BERTopic ─────────────────────────────
    embed_model = SentenceTransformer(CONFIG["bertopic_model"])
    topic_model = BERTopic(
        embedding_model=embed_model,
        language="english",
        calculate_probabilities=False,
        verbose=True
    )
    topics, _ = topic_model.fit_transform(replies)
    repr_docs = topic_model.get_representative_docs()
    logging.info("%d Cluster erstellt", len(repr_docs))

    # ─── GPT Summaries ───────────────────────
    summaries = {tid: {"summary": gpt_summary(docs), "top_docs": docs[:5]}
                 for tid, docs in repr_docs.items()}

    # ─── Export ──────────────────────────────
    out = Path(CONFIG["output_dir"]) / event_key
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(replies, name="tweet").to_csv(out / "replies.csv", index=False)
    (out / "cluster_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    topic_model.save(out / "bertopic_model")
    logging.info("Export => %s", out)

# ══════════════════════════════════════════════════════════════════
# 9  CLI + MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if not BEARER_TOKEN:
        logging.critical("❌ Umgebungsvariable TW_BEARER_TOKEN fehlt.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Moral-Misalignment – Twitter-API Pipeline")
    parser.add_argument("--events", nargs="+",
                        choices=list(CONFIG["events"].keys()) + ["ALL"],
                        default=["ALL"])
    sel = parser.parse_args().events
    selected = CONFIG["events"].keys() if "ALL" in sel else sel

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    full, pref = parse_mfd(Path(CONFIG["mfd_dic_path"]))
    random.seed(CONFIG["random_seed"])

    logging.info("Pipeline-Start %s", datetime.utcnow().isoformat(timespec="seconds"))
    for ev in selected:
        run_event(ev, nlp, full, pref)
    logging.info("Pipeline beendet.")

if __name__ == "__main__":
    main()
