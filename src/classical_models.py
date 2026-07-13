"""
Classical machine-learning models for the IMDb sentiment project.

I use this file to compare three models:

- Multinomial Naive Bayes
- Linear Support Vector Machine
- Logistic Regression

The assignment also asks to test the system with different amounts of
training data, so I repeat the experiment with 1,000, 5,000, 12,500
and 25,000 reviews.

The main order is:

1. load the cleaned IMDb reviews
2. create a balanced training subset
3. convert the text with TF-IDF
4. train the three models
5. calculate the evaluation metrics
6. save the results and figures
7. print a few mistakes for the error analysis
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
    Select the same number of positive and negative reviews.

    I first used the first rows from each class, but changed this to
    random sampling so the result does not depend on the original
    file order. The fixed seed keeps the samples reproducible.
    """

    positive = dataframe[dataframe["label"] == 1].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    negative = dataframe[dataframe["label"] == 0].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    subset = pd.concat(
        [positive, negative],
        ignore_index=True,
    )

    return (
        subset
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def create_models():
    """
    Create fresh model objects for one experiment.

    I create new models for each training-set size so that every run
    starts independently.
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
    Calculate the four metrics used in the report.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def save_confusion_matrix(y_true, y_pred, output_path, title):
    """
    Create and save a confusion matrix for the best model.
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

    # Add the number of reviews inside each square.
    for row in range(2):
        for column in range(2):
            colour = (
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
                color=colour,
                fontsize=12,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_performance_plot(results):
    """
    Plot the F1-score for each model and training-set size.

    This graph helps show whether adding more training reviews
    continues to improve the models.
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
    Print a few wrong predictions.

    I use these examples later in the report to discuss difficult
    cases such as sarcasm, mixed sentiment and contrast.
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
    Run all classical model experiments.
    """

    train_data, test_data = get_clean_frames()

    test_text = test_data["clean"]
    test_labels = test_data["label"]

    results_rows = []

    # I keep track of the best F1-score so I can save the confusion
    # matrix and error examples for the strongest model.
    best_run = {
        "f1": -1.0,
        "model": None,
        "train_size": None,
        "predictions": None,
    }

    # The values represent reviews per class:
    # 500 + 500 = 1,000 total, and so on.
    for reviews_per_class in TRAIN_SIZES_PER_CLASS:
        training_subset = balanced_subset(
            train_data,
            reviews_per_class,
        )

        train_size = len(training_subset)

        print(f"\nTraining-set size: {train_size}")

        # I create a new vectorizer for every training size.
        # It is fitted only on the training reviews.
        #
        # The test reviews are transformed afterwards with the same
        # vocabulary. This avoids using test information during training.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # individual words and word pairs
            min_df=2,             # ignore terms appearing only once
            max_features=50000,   # keep the feature matrix manageable
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

    results = pd.DataFrame(
        results_rows
    )

    results_path = RESULTS_DIR / "classical_results.csv"

    results.to_csv(
        results_path,
        index=False,
    )

    print(f"\nSaved results to: {results_path}")

    save_performance_plot(
        results
    )

    confusion_path = (
        RESULTS_DIR
        / "classical_best_confusion.png"
    )

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

    print(
        f"Saved confusion matrix to: "
        f"{confusion_path}"
    )

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
