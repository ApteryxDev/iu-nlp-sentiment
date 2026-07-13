#!/bin/bash

# Download the Stanford IMDb movie review dataset.
#
# I keep the download in a separate script so the dataset does not need
# to be uploaded to GitHub. Anyone who clones the project can run this
# file to prepare the data on their own computer.

# Stop the script if one of the commands fails.
set -e

# The script may be started from a different folder, so I first find
# the root folder of the project.
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DATA_DIR="$PROJECT_DIR/data"
DATASET_DIR="$DATA_DIR/aclImdb"
ARCHIVE_PATH="$DATA_DIR/aclImdb_v1.tar.gz"

# Official download address for the Stanford Large Movie Review Dataset.
DATASET_URL="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

echo "IMDb dataset setup"
echo "------------------"

# If the extracted dataset already exists, there is no reason to
# download and extract it again.
if [ -d "$DATASET_DIR/train/pos" ] && [ -d "$DATASET_DIR/test/neg" ]; then
    echo "The dataset is already available at:"
    echo "$DATASET_DIR"
    exit 0
fi

# Create the data folder if this is the first run.
mkdir -p "$DATA_DIR"

# Download the compressed dataset.
# curl is already available on macOS and on many Linux systems.
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "Downloading the Stanford IMDb dataset..."

    curl -L \
        "$DATASET_URL" \
        -o "$ARCHIVE_PATH"
else
    echo "The downloaded archive already exists."
    echo "Using: $ARCHIVE_PATH"
fi

# Extract the archive into the data folder.
echo "Extracting the dataset..."

tar -xzf "$ARCHIVE_PATH" \
    -C "$DATA_DIR"

# Check that the folders needed by the project were created.
if [ ! -d "$DATASET_DIR/train/pos" ] || [ ! -d "$DATASET_DIR/test/neg" ]; then
    echo "The extraction finished, but the expected folders were not found."
    echo "Please remove the archive and run the script again."
    exit 1
fi

echo
echo "Dataset setup completed successfully."
echo "Training reviews: $DATASET_DIR/train"
echo "Test reviews:     $DATASET_DIR/test"
