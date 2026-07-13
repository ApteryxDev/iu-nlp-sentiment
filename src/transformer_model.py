"""
Optional DistilBERT experiment for IMDb sentiment classification.

The main part of the project compares classical models using TF-IDF.
I added this experiment to see how a pretrained language model performs
on the same type of task.

Because DistilBERT takes much longer to train on my laptop, I use a
smaller balanced sample:

- 4,000 training reviews
- 4,000 test reviews
- 2 training epochs

This result is treated as an additional experiment rather than a direct
comparison with the classical models, which use the complete test set.
"""

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data_preprocessing import (
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    load_dataset,
)


# Settings for this experiment
MODEL_NAME = "distilbert-base-uncased"
TRAIN_SIZE = 4000
TEST_SIZE = 4000
EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 256


def choose_device():
    """
    Check which type of hardware is available.

    My Mac supports MPS, which allows PyTorch to use the Apple GPU.
    CUDA is checked as a second option, followed by the CPU.
    """

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def balanced_sample(dataframe, total_size):
    """
    Select an equal number of positive and negative reviews.

    I use random sampling rather than selecting the first rows because
    the result should not depend on the original ordering of the files.
    """

    reviews_per_class = total_size // 2

    positive = dataframe[dataframe["label"] == 1].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    negative = dataframe[dataframe["label"] == 0].sample(
        n=reviews_per_class,
        random_state=SEED,
    )

    sample = pd.concat(
        [positive, negative],
        ignore_index=True,
    )

    return (
        sample
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def calculate_metrics(y_true, y_pred):
    """
    Calculate the same metrics used for the classical models.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def trainer_metrics(eval_prediction):
    """
    Convert the raw model scores into labels for Trainer evaluation.
    """

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return calculate_metrics(
        labels,
        predictions,
    )


def save_confusion_matrix(y_true, y_pred, output_path):
    """
    Save the DistilBERT confusion matrix for the report.
    """

    import matplotlib.pyplot as plt

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    figure, axis = plt.subplots(
        figsize=(5, 4)
    )

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
    axis.set_title("DistilBERT")

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

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def make_huggingface_dataset(dataframe, tokenizer):
    """
    Convert a pandas DataFrame into the format expected by DistilBERT.
    """

    dataset = Dataset.from_pandas(
        dataframe[["text", "label"]],
        preserve_index=False,
    )

    def tokenize(batch):
        # Reviews can be longer than the model limit, so I truncate them.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    # Transformers expects the target column to be called "labels".
    dataset = dataset.rename_column(
        "label",
        "labels",
    )

    # The original review text is no longer needed after tokenisation.
    required_columns = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    columns_to_remove = [
        column
        for column in dataset.column_names
        if column not in required_columns
    ]

    return dataset.remove_columns(
        columns_to_remove
    )


def run_experiment():
    """
    Fine-tune DistilBERT and save the final evaluation results.
    """

    device = choose_device()
    print(f"Device: {device}")

    train_data, test_data = load_dataset()

    train_sample = balanced_sample(
        train_data,
        TRAIN_SIZE,
    )

    test_sample = balanced_sample(
        test_data,
        TEST_SIZE,
    )

    print(f"Training reviews: {len(train_sample)}")
    print(f"Test reviews: {len(test_sample)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = make_huggingface_dataset(
        train_sample,
        tokenizer,
    )

    test_dataset = make_huggingface_dataset(
        test_sample,
        tokenizer,
    )

    # The base DistilBERT model receives a new two-class output layer.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # Padding is performed separately for each batch.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_arguments = TrainingArguments(
        output_dir=str(
            PROJECT_ROOT / "data" / "bert_ckpt"
        ),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=trainer_metrics,
    )

    print("\nStarting DistilBERT training...")
    trainer.train()

    evaluation = trainer.evaluate()

    print("\nEvaluation results")

    for metric_name, metric_value in evaluation.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")

    # Trainer returns class scores. The class with the highest score
    # becomes the final predicted label.
    prediction_output = trainer.predict(
        test_dataset
    )

    predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    final_metrics = calculate_metrics(
        test_sample["label"],
        predictions,
    )

    results = pd.DataFrame(
        [
            {
                "model": "distilbert",
                "train_size": len(train_sample),
                "test_size": len(test_sample),
                **final_metrics,
            }
        ]
    )

    results_path = (
        RESULTS_DIR
        / "distilbert_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    confusion_path = (
        RESULTS_DIR
        / "distilbert_confusion.png"
    )

    save_confusion_matrix(
        test_sample["label"],
        predictions,
        confusion_path,
    )

    print(f"\nSaved results to: {results_path}")
    print(f"Saved confusion matrix to: {confusion_path}")


if __name__ == "__main__":
    run_experiment()
