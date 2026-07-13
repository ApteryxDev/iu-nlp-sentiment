"""
Data loading and preprocessing for the IMDb sentiment analysis project.

This file completes the first stages of the NLP pipeline:

1. Define the project paths and settings.
2. Load the original IMDb reviews from the train and test folders.
3. Clean and normalise the text.
4. Remove stop words.
5. Apply lemmatisation.
6. Save the processed data so it does not need to be cleaned again.

The resulting DataFrames contain three columns:

    text   - the original review
    label  - 1 for positive and 0 for negative
    clean  - the preprocessed review used by the machine-learning models
"""

# ============================================================
# 1. Import the required libraries
# ============================================================

from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# 2. Define project paths and reproducibility settings
# ============================================================

# The project root is the folder above the src directory.
# Using relative paths makes the project work after it is cloned
# to another computer, rather than depending on one fixed location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Location of the extracted Stanford IMDb dataset.
DATA_DIR = PROJECT_ROOT / "data" / "aclImdb"

# The cleaned reviews are stored here after the first preprocessing run.
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Tables and figures produced by the experiments are stored here.
RESULTS_DIR = PROJECT_ROOT / "results"

# A fixed random seed is used so that shuffling and sampling can be reproduced.
SEED = 42

# The assignment asks for experiments with progressively larger datasets.
# These values represent the number of reviews taken from each class.
#
# 500 per class    = 1,000 total reviews
# 2,500 per class  = 5,000 total reviews
# 6,250 per class  = 12,500 total reviews
# 12,500 per class = 25,000 total reviews
TRAIN_SIZES_PER_CLASS = [500, 2500, 6250, 12500]

# Create the folders automatically if they do not already exist.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Prepare the regular expressions used for text cleaning
# ============================================================

# IMDb reviews contain HTML tags such as <br />.
# These tags do not carry sentiment and are removed.
HTML_PATTERN = re.compile(r"<[^>]+>")

# After lowercasing, only English letters and spaces are retained.
# Numbers and punctuation are removed to simplify the representation.
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")


# ============================================================
# 4. Download the required NLTK resources when necessary
# ============================================================


def ensure_nltk_resources():
    """
    Download the NLTK resources only when they are not already installed.
    """

    required_resources = [
        ("stopwords", ["corpora/stopwords", "corpora/stopwords.zip"]),
        ("wordnet", ["corpora/wordnet", "corpora/wordnet.zip"]),
        ("punkt", ["tokenizers/punkt", "tokenizers/punkt.zip"]),
        ("punkt_tab", ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"]),
    ]

    for package_name, possible_paths in required_resources:
        resource_found = False

        for resource_path in possible_paths:
            try:
                nltk.data.find(resource_path)
                resource_found = True
                break
            except LookupError:
                continue

        if not resource_found:
            print(f"Downloading missing NLTK resource: {package_name}")
            nltk.download(package_name, quiet=True)


# ============================================================
# 5. Load one official IMDb dataset split
# ============================================================

def read_split(split: str) -> pd.DataFrame:
    """
    Read either the official training split or the official test split.

    Parameters
    ----------
    split:
        Either "train" or "test".

    Returns
    -------
    pandas.DataFrame
        A table containing the original review text and its binary label.
    """

    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")

    rows = []

    # In the IMDb dataset, positive and negative reviews are stored
    # in separate folders. Their folder name determines the label.
    label_folders = [
        ("pos", 1),
        ("neg", 0),
    ]

    for folder_name, label in label_folders:
        folder = DATA_DIR / split / folder_name

        # Give a clear error message when the dataset is unavailable.
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}\n"
                "Make sure the IMDb dataset has been downloaded and extracted."
            )

        # Each text file contains one movie review.
        # Sorting the paths keeps the initial loading order stable.
        for review_path in sorted(folder.glob("*.txt")):
            review_text = review_path.read_text(encoding="utf-8")

            rows.append(
                {
                    "text": review_text,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 6. Load the official training and test partitions
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and shuffle the official IMDb training and test sets.

    The official separation is preserved. The test reviews are never added
    to the training set, which prevents data leakage.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The shuffled training DataFrame and test DataFrame.
    """

    train_df = read_split("train")
    test_df = read_split("test")

    # Shuffling prevents the rows from remaining grouped by sentiment label.
    # The fixed seed ensures that the order is reproducible.
    train_df = (
        train_df
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    return train_df, test_df


# ============================================================
# 7. Clean and normalise one review
# ============================================================

def clean_text(
    text: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: set[str],
) -> str:
    """
    Apply the complete preprocessing procedure to one movie review.

    Processing steps:
    1. Convert the text to lowercase.
    2. Remove HTML tags.
    3. Remove punctuation and numbers.
    4. Tokenise the text into words.
    5. Remove stop words and very short tokens.
    6. Lemmatise the remaining words.
    7. Join the tokens back into a single string.

    Parameters
    ----------
    text:
        The original movie review.
    lemmatizer:
        The WordNet lemmatiser used to reduce words to their base form.
    stop_words:
        The set of English stop words to remove.

    Returns
    -------
    str
        The cleaned review.
    """

    # Lowercasing ensures that words such as "Good" and "good"
    # are treated as the same feature.
    text = text.lower()

    # Remove HTML formatting left over from the original IMDb pages.
    text = HTML_PATTERN.sub(" ", text)

    # Remove punctuation, digits, and other non-letter characters.
    text = NON_ALPHA_PATTERN.sub(" ", text)

    # Split the review into individual word tokens.
    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:
        # Common stop words are removed because they often carry little
        # information for document classification.
        if token in stop_words:
            continue

        # Very short tokens are removed because many are fragments or noise.
        if len(token) <= 2:
            continue

        # Lemmatise the token to reduce related forms to a common base.
        lemma = lemmatizer.lemmatize(token)
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


# ============================================================
# 8. Preprocess the full dataset and cache the result
# ============================================================

def get_clean_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return cleaned training and test DataFrames.

    If cached copies already exist, they are loaded directly. Otherwise,
    the original text is cleaned and the processed DataFrames are saved.

    Caching is useful because preprocessing all 50,000 reviews takes much
    longer than loading the previously processed files.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The cleaned training and test DataFrames.
    """

    train_cache_path = CACHE_DIR / "train_clean.parquet"
    test_cache_path = CACHE_DIR / "test_clean.parquet"

    # Reuse previously cleaned data when both cache files are available.
    if train_cache_path.exists() and test_cache_path.exists():
        print("Loading cleaned reviews from cache.")

        train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)

        return train_df, test_df

    print("No complete cache found. Preprocessing the IMDb reviews.")

    # Ensure that tokenisation, stop-word removal, and lemmatisation
    # have access to their required language resources.
    ensure_nltk_resources()

    train_df, test_df = load_dataset()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Apply exactly the same cleaning process to the training and test sets.
    # The labels remain unchanged.
    train_df["clean"] = train_df["text"].apply(
        lambda review: clean_text(
            review,
            lemmatizer,
            stop_words,
        )
    )

    test_df["clean"] = test_df["text"].apply(
        lambda review: clean_text(
            review,
            lemmatizer,
            stop_words,
        )
    )

    # Save the processed reviews for future experiment runs.
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    print(f"Saved training cache to: {train_cache_path}")
    print(f"Saved test cache to: {test_cache_path}")

    return train_df, test_df


# ============================================================
# 9. Run a simple validation check when this file is executed
# ============================================================

if __name__ == "__main__":
    train_data, test_data = get_clean_frames()

    print("\nDataset validation")
    print("------------------")
    print(f"Training reviews: {len(train_data)}")
    print(f"Test reviews:     {len(test_data)}")
    print(
        "Training labels:",
        train_data["label"].value_counts().sort_index().to_dict(),
    )
    print(
        "Test labels:",
        test_data["label"].value_counts().sort_index().to_dict(),
    )

    print("\nExample of an original review:")
    print(train_data.iloc[0]["text"][:300])

    print("\nThe same review after preprocessing:")
    print(train_data.iloc[0]["clean"][:300])
