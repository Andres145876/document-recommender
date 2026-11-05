# Top-K Document Recommender with Cosine Similarity

## Objective

Implement a Python script that, given a document ID or a short free-text query, recommends the **5 most similar documents** in a repository using **TF-IDF and cosine similarity** (Path A), and validate its results and its business utility (less rework, faster decisions).

The corpus is simulated: 30 documents from a fictional IT services company ("Altavista") in four categories: Policy/Process (10), Support/Incidents (10), FAQ (5), and Minutes (5).

## File structure

```
document-recommender/
├── recommender.py         # Source code (loading, cleaning, TF-IDF, cosine, interface)
├── corpus.csv             # Dataset: id, title, text, category (30 documents)
├── requirements.txt       # Dependencies
├── README.md
├── report.pdf             # 1-2 page report with the video link
└── evidencias/            # Evidence of the tests
    ├── test1_policy_process.txt       # Test 1: seed DOC001
    ├── test2_support_incidents.txt    # Test 2: seed DOC012
    ├── test3_free_text.txt            # Test 3: free-text query
    ├── extra_limitation_example.txt   # Extra: example of a limitation
    └── category_evaluation.txt        # Extra: coherence by category
```

## Dependencies

- Python 3.9 or higher
- scikit-learn and numpy (see `requirements.txt`)

## Execution steps (Windows)

Run everything from the folder that contains `recommender.py`.

1. Create and activate a virtual environment.

   Command Prompt (cmd):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```
   PowerShell:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   (If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)
2. Install the dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Run the recommender (see the usage modes below). No file needs to be modified.

On macOS/Linux use `python3 -m venv .venv`, `source .venv/bin/activate`, and `python3 recommender.py ...`.

## Usage modes

| Command | What it does |
|---|---|
| `python recommender.py` | Interactive menu (ID or free text) |
| `python recommender.py --id DOC001` | Top-5 from a seed document |
| `python recommender.py --query "text"` | Top-5 from a free-text query |
| `python recommender.py --list` | Show the available IDs, categories, and titles |
| `python recommender.py --evaluate` | Measure what fraction of the Top-5 shares the seed's category |

Extra options: `--k N` (number of recommendations, default 5), `--corpus PATH` (another CSV file), and `--save FILE` (also save the output to a UTF-8 .txt file, used to generate the evidence).

## Usage example

```cmd
python recommender.py --id DOC012
```

Output:

```
INPUT (ID): DOC012 | Support/Incidents | Ticket: VPN will not connect from home, authentication error
Top-5 recommended documents (cosine similarity)
#   ID       Score    Category             Title
1   DOC022   0.5002   FAQ                  FAQ: How do I connect to the VPN?
2   DOC003   0.3073   Policy/Process       VPN and Remote Work Policy
3   DOC001   0.1545   Policy/Process       Password and Multi-Factor Authentication Policy
4   DOC011   0.1417   Support/Incidents    Ticket: User locked out after failed password attempts
5   DOC021   0.1304   FAQ                  FAQ: How do I reset my password?
```

With free text:

```cmd
python recommender.py --query "My account is locked after I changed my password"
```

## How it works

1. **Loading and validation** (`load_corpus`): checks that the file exists, that the columns `id`, `title`, and `text` are present, that there are no empty or duplicated IDs, and that no text is empty.
2. **Cleaning** (`clean_text`): lowercase, removal of accents and symbols, English stopword removal, and light stemming (plurals and endings such as -ing / -ed).
3. **Vectorization**: `TfidfVectorizer` over the title + text of each document.
4. **Similarity**: `cosine_similarity` between the seed (or the query) and the whole corpus.
5. **Top-5**: scores are sorted from highest to lowest; when an ID is used, the seed document is excluded.

## Error handling

The program prints a clear message (it does not crash) for: a non-existent ID (it suggests similar IDs), an empty ID or query, a query with no words present in the corpus, a missing corpus file, missing columns, and duplicated IDs.

## Evidence

The three required tests are in `/evidencias/`. Each file includes the command, the input, the Top-5 with similarity scores, and a comment on whether the recommendations are coherent:

1. `test1_policy_process.txt`: seed DOC001 (Policy/Process)
2. `test2_support_incidents.txt`: seed DOC012 (Support/Incidents)
3. `test3_free_text.txt`: free-text query

To reproduce the outputs:

```cmd
python recommender.py --id DOC001 --save evidencias/test1_policy_process.txt
python recommender.py --id DOC012 --save evidencias/test2_support_incidents.txt
python recommender.py --query "My account is locked after I changed my password" --save evidencias/test3_free_text.txt
```

(The command writes the recommender output; the comment in each file was written separately and will be overwritten if you rerun it.)

## Reflection video

Link: _PASTE THE VIDEO LINK HERE_
