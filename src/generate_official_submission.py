import json
import difflib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR = MODELS_DIR / "submitted_t5_base_checkpoint29640"

TRAIN_PATH = DATA_DIR / "train.json"
VALID_PATH = DATA_DIR / "valid.json"
TEST_PATH = DATA_DIR / "test.json"
ARASAAC_PATH = ROOT_DIR / "resources" / "arasaac_english.json"

RAW_TEST_PREDICTIONS_PATH = OUTPUT_DIR / "raw_test_predictions_greedy.json"
POSTPROCESSED_TEST_PATH = OUTPUT_DIR / "postprocessed_test_predictions_greedy.json"
SUBMISSION_PATH = OUTPUT_DIR / "official_submission.json"

USE_PREFIX = True
MAX_SOURCE_LENGTH = 64
MAX_NEW_TOKENS = 32
BATCH_SIZE = 16

FUZZY_THRESHOLD = 0.92
MIN_TOKEN_FREQ_FOR_FUZZY_TARGET = 2


# =========================
# UTILS
# =========================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def tokenize(text):
    return str(text).strip().split()


def detokenize(tokens):
    return " ".join(tokens)


def normalize_src(text):
    text = str(text).strip()
    if USE_PREFIX:
        return "translate English to pictograms: " + text
    return text


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_arasaac_term(term):
    return str(term).strip().replace(" ", "_")


# =========================
# VOCAB + POSTPROCESS
# =========================

def build_competition_maps(rows):
    comp_vocab = Counter()
    token_to_pictos = defaultdict(Counter)
    picto_to_tokens = defaultdict(Counter)

    skipped = 0

    for row in rows:
        tgt = row.get("tgt", "")
        pictos = row.get("pictos", [])

        tokens = tokenize(tgt)

        if len(tokens) != len(pictos):
            skipped += 1
            continue

        for tok, picto in zip(tokens, pictos):
            picto = str(picto)
            comp_vocab[tok] += 1
            token_to_pictos[tok][picto] += 1
            picto_to_tokens[picto][tok] += 1

    print(f"Competition vocabulary: {len(comp_vocab)} tokens únicos")
    print(f"Rows skipped due to tgt/pictos misalignment: {skipped}")

    return comp_vocab, picto_to_tokens


def build_arasaac_maps(arasaac_rows):
    norm_term_to_ids = defaultdict(set)

    for item in arasaac_rows:
        picto_id = str(item.get("_id"))

        for kw in item.get("keywords", []):
            for field in ["keyword", "plural"]:
                term = kw.get(field)
                if not term:
                    continue

                norm = normalize_arasaac_term(term)
                norm_term_to_ids[norm].add(picto_id)

    return norm_term_to_ids


def build_arasaac_to_competition_corrections(comp_vocab, picto_to_tokens, norm_term_to_ids):
    corrections = {}

    for norm_term, picto_ids in norm_term_to_ids.items():
        candidate_tokens = Counter()

        for picto_id in picto_ids:
            if picto_id in picto_to_tokens:
                candidate_tokens.update(picto_to_tokens[picto_id])

        if candidate_tokens:
            best_token, best_count = candidate_tokens.most_common(1)[0]

            if best_token != norm_term:
                corrections[norm_term] = {
                    "replacement": best_token,
                    "method": "arasaac_picto_to_seen_competition_token",
                    "candidate_count": best_count,
                    "picto_ids": sorted(list(picto_ids))[:20],
                }

    print(f"ARASAAC-to-competition corrections: {len(corrections)}")
    return corrections


def should_skip_fuzzy(tok):
    if not tok:
        return True
    if any(c.isupper() for c in tok):
        return True
    if len(tok) <= 3:
        return True
    return False


def get_fuzzy_replacement(tok, candidate_vocab):
    matches = difflib.get_close_matches(
        tok,
        candidate_vocab,
        n=1,
        cutoff=FUZZY_THRESHOLD,
    )
    return matches[0] if matches else None


def postprocess_predictions(predictions, comp_vocab, arasaac_corrections):
    candidate_vocab = [
        tok for tok, freq in comp_vocab.items()
        if freq >= MIN_TOKEN_FREQ_FOR_FUZZY_TARGET
    ]

    processed = []
    changes = []
    unknown_before = Counter()
    unknown_after = Counter()

    for pred in predictions:
        old_tokens = tokenize(pred["hyp"])
        new_tokens = []
        row_changes = []

        for tok in old_tokens:
            if tok in comp_vocab:
                new_tokens.append(tok)
                continue

            unknown_before[tok] += 1

            replacement = None
            method = None

            if tok in arasaac_corrections:
                replacement = arasaac_corrections[tok]["replacement"]
                method = arasaac_corrections[tok]["method"]

            if replacement is None and not should_skip_fuzzy(tok):
                fuzzy = get_fuzzy_replacement(tok, candidate_vocab)
                if fuzzy is not None:
                    replacement = fuzzy
                    method = "high_confidence_fuzzy_match"

            if replacement is not None and replacement in comp_vocab:
                new_tokens.append(replacement)
                row_changes.append({
                    "old": tok,
                    "new": replacement,
                    "method": method,
                })
            else:
                new_tokens.append(tok)
                unknown_after[tok] += 1

        new_hyp = detokenize(new_tokens)

        new_pred = dict(pred)
        new_pred["hyp_original"] = pred["hyp"]
        new_pred["hyp"] = new_hyp
        new_pred["postprocess_changes"] = row_changes
        processed.append(new_pred)

        if row_changes:
            changes.append({
                "id": pred.get("id"),
                "src": pred.get("src"),
                "hyp_before": pred.get("hyp"),
                "hyp_after": new_hyp,
                "changes": row_changes,
            })

    print("Postprocessing summary:")
    print(json.dumps({
        "num_predictions": len(predictions),
        "num_predictions_changed": len(changes),
        "num_token_changes": sum(len(x["changes"]) for x in changes),
        "unknown_before_total": sum(unknown_before.values()),
        "unknown_before_unique": len(unknown_before),
        "unknown_after_total": sum(unknown_after.values()),
        "unknown_after_unique": len(unknown_after),
        "top_unknown_before": unknown_before.most_common(30),
        "top_unknown_after": unknown_after.most_common(30),
    }, ensure_ascii=False, indent=2))

    return processed, changes


# =========================
# GENERATION
# =========================

def generate_test_predictions(test_rows, tokenizer, model, device):
    predictions = []

    with torch.no_grad():
        for batch_rows in tqdm(
            list(batched(test_rows, BATCH_SIZE)),
            desc="Generating test predictions with greedy decoding"
        ):
            input_texts = [
                normalize_src(row["src"])
                for row in batch_rows
            ]

            encoded = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=1,
                do_sample=False,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )

            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for row, hyp in zip(batch_rows, decoded):
                predictions.append({
                    "id": row["id"],
                    "src": row["src"],
                    "hyp": hyp.strip(),
                })

    return predictions


def validate_submission(submission, test_rows):
    test_ids = [row["id"] for row in test_rows]
    sub_ids = [row["id"] for row in submission]

    problems = {
        "num_test": len(test_rows),
        "num_submission": len(submission),
        "missing_ids": len(set(test_ids) - set(sub_ids)),
        "extra_ids": len(set(sub_ids) - set(test_ids)),
        "duplicate_ids": len(sub_ids) - len(set(sub_ids)),
        "empty_hyp": sum(1 for row in submission if not str(row.get("hyp", "")).strip()),
        "bad_keys": sum(1 for row in submission if set(row.keys()) != {"id", "hyp"}),
    }

    print("Submission validation:")
    print(json.dumps(problems, ensure_ascii=False, indent=2))

    if any(problems[k] for k in ["missing_ids", "extra_ids", "duplicate_ids", "empty_hyp", "bad_keys"]):
        raise RuntimeError("ERROR: Submission con problemas. Revisa el resumen anterior.")

    print("Submission is valid")


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_DIR}")

    print(f"Loading submitted checkpoint from: {MODEL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    print("Loading data...")

    train_rows = load_json(TRAIN_PATH)
    valid_rows = load_json(VALID_PATH)
    test_rows = load_json(TEST_PATH)
    arasaac_rows = load_json(ARASAAC_PATH)

    print(f"Train: {len(train_rows)}")
    print(f"Valid: {len(valid_rows)}")
    print(f"Test: {len(test_rows)}")

    comp_vocab, picto_to_tokens = build_competition_maps(train_rows + valid_rows)

    norm_term_to_ids = build_arasaac_maps(arasaac_rows)
    arasaac_corrections = build_arasaac_to_competition_corrections(
        comp_vocab=comp_vocab,
        picto_to_tokens=picto_to_tokens,
        norm_term_to_ids=norm_term_to_ids,
    )

    raw_predictions = generate_test_predictions(
        test_rows=test_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    save_json(RAW_TEST_PREDICTIONS_PATH, raw_predictions)

    postprocessed_predictions, changes = postprocess_predictions(
        predictions=raw_predictions,
        comp_vocab=comp_vocab,
        arasaac_corrections=arasaac_corrections,
    )

    save_json(POSTPROCESSED_TEST_PATH, postprocessed_predictions)

    submission = [
        {
            "id": row["id"],
            "hyp": row["hyp"],
        }
        for row in postprocessed_predictions
    ]

    validate_submission(submission, test_rows)

    save_json(SUBMISSION_PATH, submission)

    print("\nOfficial submission generated")
    print(json.dumps({
        "model_dir": str(MODEL_DIR),
        "raw_predictions": str(RAW_TEST_PREDICTIONS_PATH),
        "postprocessed_predictions": str(POSTPROCESSED_TEST_PATH),
        "submission": str(SUBMISSION_PATH),
        "num_predictions": len(submission),
        "decoding": {
            "name": "greedy_lp1.0",
            "num_beams": 1,
            "length_penalty": 1.0,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
