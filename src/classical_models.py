"""
Classical machine learning experiments for IMDb sentiment analysis.

The script compares three supervised models:

- Multinomial Naive Bayes
- Linear Support Vector Machine
- Logistic Regression

Each model is trained with four different amounts of labelled data.
The results are evaluated on the same 25,000-review test set so that
the models and training sizes can be compared fairly.
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .data_preprocessing import (
    RESULTS_DIR,
    SEED,
    TRAIN_SIZES_PER_CLASS,
    get_clean_frames,
)


def balanced_subset(dataframe, reviews_per_class):
    """
    Return a random subset with equal positive and negative reviews.

    A fixed random seed is used so that the same samples can be
    selected again when the experiment is repeated.
    """

    positive_reviews = dataframe[dataframe["label"] == 1].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    negative_reviews = dataframe[dataframe["label"] == 0].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    subset = pd.concat(
        [positive_reviews, negative_reviews],
        ignore_index=True,
    )

    # Shuffle the two classes together before model training.
    return subset.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)


def create_models():
    """
    Create new model objects for one training-size experiment.

    New instances are used so that each experiment begins with
    models that have not already been fitted.
    """

    return {
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(
            random_state=SEED,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        ),
    }


def calculate_metrics(y_true, y_pred):
    """
    Calculate the evaluation metrics reported in the project.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def save_confusion_matrix(y_true, y_pred, output_path, title):
    """
    Save a confusion matrix for the strongest classical model.
    """

    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(5, 4))

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    axis.set_xticks(
        [0, 1],
        labels=["Negative", "Positive"],
    )
    axis.set_yticks(
        [0, 1],
        labels=["Negative", "Positive"],
    )

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(title)

    # Display the number of reviews inside each matrix cell.
    for row in range(2):
        for column in range(2):
            text_colour = (
                "white"
                if matrix[row, column] > matrix.max() / 2
                else "black"
            )

            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=text_colour,
                fontsize=12,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_performance_plot(results):
    """
    Plot the F1-score of each model at every training-set size.
    """

    output_path = RESULTS_DIR / "performance_by_size.png"

    plt.figure(figsize=(8, 5))

    for model_name, model_results in results.groupby("model"):
        model_results = model_results.sort_values("train_size")
        readable_name = model_name.replace("_", " ").title()

        plt.plot(
            model_results["train_size"],
            model_results["f1"],
            marker="o",
            label=readable_name,
        )

    plt.xlabel("Number of training reviews")
    plt.ylabel("F1-score")
    plt.title("Model performance by training-set size")

    plt.xticks(
        sorted(results["train_size"].unique())
    )
    plt.ylim(0.79, 0.90)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved performance graph to: {output_path}")


def print_error_examples(test_data, predictions, number_of_examples=5):
    """
    Print examples of false-positive and false-negative predictions.

    These examples are useful for examining cases that are difficult
    for a TF-IDF model, such as sarcasm and mixed opinions.
    """

    errors = test_data[["text", "label"]].copy()
    errors["prediction"] = predictions

    false_positives = errors[
        (errors["label"] == 0)
        & (errors["prediction"] == 1)
    ]

    false_negatives = errors[
        (errors["label"] == 1)
        & (errors["prediction"] == 0)
    ]

    total_errors = len(false_positives) + len(false_negatives)

    print(
        f"\nTotal misclassified reviews: "
        f"{total_errors} of {len(test_data)}"
    )

    print("\nFalse positives")
    print("----------------")

    for review in false_positives["text"].head(number_of_examples):
        print(f"- {review[:500]}\n")

    print("\nFalse negatives")
    print("----------------")

    for review in false_negatives["text"].head(number_of_examples):
        print(f"- {review[:500]}\n")


def run_experiment():
    """
    Train and evaluate all classical models.
    """

    train_data, test_data = get_clean_frames()

    test_text = test_data["clean"]
    test_labels = test_data["label"]

    results_rows = []

    # Store the strongest result so its confusion matrix and error
    # examples can be produced after all experiments are complete.
    best_run = {
        "f1": -1.0,
        "model": None,
        "train_size": None,
        "predictions": None,
    }

    # Each value represents the number of reviews taken from one class.
    # For example, 500 positive and 500 negative reviews give 1,000 total.
    for reviews_per_class in TRAIN_SIZES_PER_CLASS:
        training_subset = balanced_subset(
            train_data,
            reviews_per_class,
        )

        train_size = len(training_subset)

        print(f"\nTraining-set size: {train_size}")

        # A new vectorizer is fitted for each training size.
        # The test set is transformed only after the training vocabulary
        # has been learned, which prevents information leakage.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )

        training_features = vectorizer.fit_transform(
            training_subset["clean"]
        )

        test_features = vectorizer.transform(
            test_text
        )

        for model_name, model in create_models().items():
            model.fit(
                training_features,
                training_subset["label"],
            )

            predictions = model.predict(
                test_features
            )

            metrics = calculate_metrics(
                test_labels,
                predictions,
            )

            results_rows.append(
                {
                    "model": model_name,
                    "train_size": train_size,
                    **metrics,
                }
            )

            print(
                f"{model_name:22s} "
                f"accuracy={metrics['accuracy']:.4f}  "
                f"precision={metrics['precision']:.4f}  "
                f"recall={metrics['recall']:.4f}  "
                f"f1={metrics['f1']:.4f}"
            )

            if metrics["f1"] > best_run["f1"]:
                best_run = {
                    "model": model_name,
                    "train_size": train_size,
                    "f1": metrics["f1"],
                    "predictions": predictions,
                }

    results = pd.DataFrame(results_rows)

    results_path = RESULTS_DIR / "classical_results.csv"

    results.to_csv(
        results_path,
        index=False,
    )

    print(f"\nSaved results to: {results_path}")

    save_performance_plot(results)

    confusion_path = RESULTS_DIR / "classical_best_confusion.png"

    model_title = (
        best_run["model"]
        .replace("_", " ")
        .title()
    )

    save_confusion_matrix(
        test_labels,
        best_run["predictions"],
        confusion_path,
        title=(
            f"{model_title} "
            f"(n={best_run['train_size']})"
        ),
    )

    print(f"Saved confusion matrix to: {confusion_path}")

    print(
        f"\nBest model: "
        f"{best_run['model']} "
        f"with {best_run['train_size']} reviews "
        f"and F1={best_run['f1']:.4f}"
    )

    print_error_examples(
        test_data,
        best_run["predictions"],
    )


if __name__ == "__main__":
    run_experiment()
