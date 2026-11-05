import argparse
import csv
import difflib
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.csv"
REQUIRED_COLUMNS = {"id", "title", "text"}
TOP_K = 5

# Common English stopwords (removed during text cleaning).
STOPWORDS_EN = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "if",
    "in", "into", "is", "it", "its", "me", "may", "must", "my", "no", "not", "of",
    "on", "once", "only", "or", "other", "our", "she", "so", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "up", "was", "we", "were", "what", "when", "which", "who", "will",
    "with", "within", "without", "would", "you", "your", "should", "being",
    "each", "every", "one", "two",
}


# --------------------------------------------------------------------------
# 1. Corpus loading and validation
# --------------------------------------------------------------------------
def load_corpus(path):
    """Read corpus.csv and return a list of dicts (id, title, text, category).

    Raises FileNotFoundError if the file does not exist and ValueError if
    columns are missing, IDs are empty or duplicated, or a text is empty.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"The corpus is missing the columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    documents, seen = [], set()
    for number, row in enumerate(rows, start=2):  # row 1 is the header
        doc_id = (row.get("id") or "").strip()
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not doc_id:
            raise ValueError(f"Row {number}: the document has no id.")
        if doc_id in seen:
            raise ValueError(f"Row {number}: the id '{doc_id}' is duplicated.")
        if not text:
            raise ValueError(f"Row {number} ({doc_id}): the text is empty.")
        seen.add(doc_id)
        documents.append({
            "id": doc_id,
            "title": title or "(untitled)",
            "text": text,
            "category": (row.get("category") or "").strip() or "Uncategorized",
        })

    if len(documents) < 2:
        raise ValueError("The corpus needs at least 2 documents.")
    return documents


# --------------------------------------------------------------------------
# 2. Text cleaning
# --------------------------------------------------------------------------
def _simple_stem(token):
    """Very light stemming so that 'passwords'/'password' or
    'changing'/'changed'/'change' end up as the same term."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def clean_text(text):
    """Minimal cleaning: lowercase, no accents or symbols, no stopwords, light stemming."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = []
    for token in text.split():
        if len(token) < 2 or token in STOPWORDS_EN:
            continue
        tokens.append(_simple_stem(token))
    return " ".join(tokens)


# --------------------------------------------------------------------------
# 3. Recommender
# --------------------------------------------------------------------------
class Recommender:
    """Vectorizes the corpus with TF-IDF and computes cosine similarity."""

    def __init__(self, documents):
        self.documents = documents
        self.index_by_id = {d["id"].upper(): i for i, d in enumerate(documents)}
        # Title + text are concatenated: the title adds topic keywords.
        cleaned = [clean_text(f"{d['title']} {d['text']}") for d in documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(cleaned)

    def _top_k(self, similarities, k, exclude=None):
        """Sort similarities from highest to lowest and return the first k."""
        similarities = similarities.copy()
        if exclude is not None:
            similarities[exclude] = -1.0  # the seed document is never recommended
        k = max(1, min(k, len(self.documents) - (1 if exclude is not None else 0)))
        order = np.argsort(-similarities, kind="stable")[:k]
        return [(int(i), float(similarities[i])) for i in order]

    def recommend_by_id(self, doc_id, k=TOP_K):
        """Top-k documents most similar to a document in the corpus."""
        key = (doc_id or "").strip().upper()
        if not key:
            raise ValueError("The document ID is empty.")
        if key not in self.index_by_id:
            close = difflib.get_close_matches(key, self.index_by_id, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"The ID '{doc_id}' does not exist in the corpus.{hint}")
        idx = self.index_by_id[key]
        similarities = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        return self._top_k(similarities, k, exclude=idx)

    def recommend_by_text(self, query, k=TOP_K):
        """Top-k documents most similar to a free-text query."""
        if not query or not query.strip():
            raise ValueError("The query is empty. Type at least one word.")
        cleaned = clean_text(query)
        if not cleaned:
            raise ValueError("The query only contains stopwords or symbols.")
        vector = self.vectorizer.transform([cleaned])
        if vector.nnz == 0:
            raise ValueError("None of the query words appear in the corpus; try other terms.")
        similarities = cosine_similarity(vector, self.matrix).ravel()
        return self._top_k(similarities, k)


# --------------------------------------------------------------------------
# 4. Output formatting
# --------------------------------------------------------------------------
def _shorten(text, length):
    return text if len(text) <= length else text[: length - 3] + "..."


def format_results(header, results, documents):
    """Build the text table with the Top-K and its similarity scores."""
    lines = [
        "=" * 100,
        header,
        "=" * 100,
        f"Top-{len(results)} recommended documents (cosine similarity)",
        f"{'#':<3} {'ID':<8} {'Score':<8} {'Category':<20} Title",
        "-" * 100,
    ]
    for position, (idx, score) in enumerate(results, start=1):
        doc = documents[idx]
        lines.append(
            f"{position:<3} {doc['id']:<8} {score:<8.4f} "
            f"{_shorten(doc['category'], 19):<20} {_shorten(doc['title'], 60)}"
        )
    return "\n".join(lines)


def header_for_id(recommender, doc_id):
    doc = recommender.documents[recommender.index_by_id[doc_id.strip().upper()]]
    return f"INPUT (ID): {doc['id']} | {doc['category']} | {doc['title']}"


def list_documents(documents):
    lines = [f"{'ID':<8} {'Category':<20} Title", "-" * 90]
    for d in documents:
        lines.append(f"{d['id']:<8} {_shorten(d['category'], 19):<20} {_shorten(d['title'], 60)}")
    return "\n".join(lines)


def evaluate_by_category(recommender, k=TOP_K):
    """Use every document as a seed and measure what fraction of its Top-k
    belongs to the same category. A rough validation of the results."""
    docs = recommender.documents
    by_category = {}
    for i, doc in enumerate(docs):
        sims = cosine_similarity(recommender.matrix[i], recommender.matrix).ravel()
        top = recommender._top_k(sims, k, exclude=i)
        hits = sum(docs[j]["category"] == doc["category"] for j, _ in top)
        by_category.setdefault(doc["category"], []).append(hits / len(top))
    lines = [
        "Category coherence (fraction of the Top-5 that shares the seed's category)",
        "-" * 90,
    ]
    everything = []
    for category, values in by_category.items():
        everything.extend(values)
        lines.append(f"{category:<22} documents: {len(values):<3} average: {np.mean(values):.2f}")
    lines.append(f"{'OVERALL':<22} documents: {len(everything):<3} average: {np.mean(everything):.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 5. Command-line interface
# --------------------------------------------------------------------------
def interactive_mode(recommender, k):
    """Simple menu to try the recommender without remembering the arguments."""
    print("Document recommender (choose 3 to exit)")
    while True:
        print("\n1) Search by document ID\n2) Search by free text\n3) Exit")
        option = input("Choose an option: ").strip()
        try:
            if option == "1":
                doc_id = input("Document ID (e.g. DOC001): ")
                results = recommender.recommend_by_id(doc_id, k)
                print(format_results(header_for_id(recommender, doc_id), results, recommender.documents))
            elif option == "2":
                query = input("Type your text: ")
                results = recommender.recommend_by_text(query, k)
                print(format_results(f"INPUT (TEXT): {query.strip()}", results, recommender.documents))
            elif option == "3":
                print("Goodbye.")
                return
            else:
                print("Invalid option. Choose 1, 2, or 3.")
        except ValueError as error:
            print(f"Error: {error}")


def build_parser():
    parser = argparse.ArgumentParser(description="Top-K document recommender (TF-IDF + cosine similarity).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--id", help="Seed document ID (e.g. DOC001)")
    group.add_argument("--query", "-q", help="Free text used to find similar documents")
    group.add_argument("--list", action="store_true", help="Show the available documents")
    group.add_argument("--evaluate", action="store_true", help="Measure Top-5 coherence by category")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Path to the corpus.csv file")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of recommendations (default 5)")
    parser.add_argument("--save", help="Also save the output to this text file (UTF-8)")
    return parser


def main(argv=None):
    # Avoid encoding errors in the Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    if args.k < 1:
        print("Error: --k must be a positive integer.", file=sys.stderr)
        return 1

    try:
        documents = load_corpus(args.corpus)
        recommender = Recommender(documents)

        if args.list:
            output = list_documents(documents)
        elif args.evaluate:
            output = evaluate_by_category(recommender, args.k)
        elif args.id is not None:
            results = recommender.recommend_by_id(args.id, args.k)
            output = format_results(header_for_id(recommender, args.id), results, documents)
        elif args.query is not None:
            results = recommender.recommend_by_text(args.query, args.k)
            output = format_results(f"INPUT (TEXT): {args.query.strip()}", results, documents)
        else:
            interactive_mode(recommender, args.k)
            return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(output)
    if args.save:
        target = Path(args.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
        print(f"\n[Output saved to {target}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
