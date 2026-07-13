"""
Use the final Logistic Regression model on new movie reviews.

The main experiments compare several models, but Logistic Regression gave
the best result with the complete training set. I therefore use it here
to demonstrate the final sentiment-analysis system.

The program first trains the model on the cleaned IMDb training reviews.
It then accepts a new review and predicts either positive or negative.

Run interactively:

    python -m src.predict

Or give the review directly:

    python -m src.predict "The film was excellent."
"""

import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data_preprocessing import (
    SEED,
    clean_text,
    ensure_nltk_resources,
    get_clean_frames,
)


def train_final_model():
    """
    Train Logistic Regression using all 25,000 training reviews.

    I use the same TF-IDF settings as in the main experiment so that
    the demonstration follows the same methodology.
    """

    train_data, _ = get_clean_frames()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    training_features = vectorizer.fit_transform(
        train_data["clean"]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    model.fit(
        training_features,
        train_data["label"],
    )

    return vectorizer, model


def predict_review(
    review,
    vectorizer,
    model,
    lemmatizer,
    stop_words,
):
    """
    Clean one new review and return its predicted sentiment.
    """

    cleaned_review = clean_text(
        review,
        lemmatizer,
        stop_words,
    )

    review_features = vectorizer.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_features
    )[0]

    if prediction == 1:
        return "positive"

    return "negative"


def main():
    """
    Train the model and allow the user to test new reviews.
    """

    print("Training the final Logistic Regression model...")

    vectorizer, model = train_final_model()

    ensure_nltk_resources()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # When a review is supplied after the command, predict it once
    # and then finish.
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])

        sentiment = predict_review(
            review,
            vectorizer,
            model,
            lemmatizer,
            stop_words,
        )

        print(f"Predicted sentiment: {sentiment}")
        print(f"Review: {review}")
        return

    # Without a command-line review, the program stays open so that
    # several examples can be tested.
    print("\nType a movie review.")
    print("Enter 'quit' when finished.\n")

    while True:
        review = input("> ").strip()

        if review.lower() in {"quit", "exit"}:
            break

        if not review:
            print("Please enter a review.")
            continue

        sentiment = predict_review(
            review,
            vectorizer,
            model,
            lemmatizer,
            stop_words,
        )

        print(f"Prediction: {sentiment}\n")


if __name__ == "__main__":
    main()
