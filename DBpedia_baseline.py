#!/usr/bin/python3

"""
DBpedia Ontology Dataset - V1 Baseline.

This stage evaluates the four baseline combinations:

* CountVectorizer + MultinomialNB
* TF-IDF + MultinomialNB
* CountVectorizer + LinearSVC
* TF-IDF + LinearSVC

The experiments are run on nested training sets of:

50, 200, 500, 2000, and 10000 observations.

All models are evaluated on the same fixed DBpedia test set.
"""

from DBpedia_framework import (
    TRAIN_SIZES,
    add_stage_metadata,
    generate_baseline_configs,
    run_experiments,
    save_stage_csv,
)

from plots import plot_confusion_matrices

"""
How does the amount of training data affect the performance
of different text representations and classifiers,
particularly in low-resource settings?
"""

# ---------------------------------------------------------------------------
# Baseline configurations
# ---------------------------------------------------------------------------

VECTORIZER_SPECS = [
    {
        "name": "Count",
        "variant": "Baseline",
    },
    {
        "name": "TF-IDF",
        "variant": "Baseline",
    },
]

CLASSIFIER_SPECS = [
    {
        "name": "MultinomialNB",
    },
    {
        "name": "LinearSVC",
    },
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Run the V1 baseline experiments on DBpedia.

    The shared stage framework handles:

    1. Loading the fixed DBpedia test set.
    2. Loading each DBpedia training set.
    3. Building the text-classification pipelines.
    4. Training and evaluating all configurations.
    5. Measuring Accuracy and Macro F1.
    6. Measuring training and inference times.
    """

    print()

    print("=" * 80)
    print("V1 BASELINE - DBPEDIA ONTOLOGY DATASET")
    print("=" * 80)

    print()

    print(
        "Training sizes: "
        f"{TRAIN_SIZES}"
    )

    print()

    # -------------------------------------------------
    # Generate baseline configurations
    # -------------------------------------------------

    configs = generate_baseline_configs(
        vectorizer_specs=VECTORIZER_SPECS,
        classifier_specs=CLASSIFIER_SPECS,
    )

    print(
        f"Number of experiments: "
        f"{len(configs)}"
    )

    # -------------------------------------------------
    # Run experiments
    # -------------------------------------------------

    results_df = run_experiments(configs)

    if results_df is None:

        print("No results were generated")

        return

    if results_df.empty:

        print("The results DataFrame is empty")
        return

    # -------------------------------------------------
    # Add shared metadata
    # -------------------------------------------------

    results_df = add_stage_metadata(
        results_df,
        experiment_name="V1",
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print()
    print("=" * 80)
    print("V1 DBPEDIA RESULTS")
    print("=" * 80)

    display_columns = [

        "Experiment",
        "Train size",
        "Variant",
        "Base Vectorizer",
        "StopWords",
        "Preprocessor",
        "Classifier",
        "Accuracy",
        "Macro F1",
        "Train time (s)",
        "Inference time (s)",
    ]

    print(
        results_df[
            display_columns
        ].round(
            {
                "Accuracy": 4,
                "Macro F1": 4,
                "Train time (s)": 4,
                "Inference time (s)": 4,
            }
        )
    )

    # -------------------------------------------------
    # Confusion matrices
    # -------------------------------------------------

    plot_confusion_matrices(results_df)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    csv_columns = [
        "Experiment",
        "Train size",
        "Variant",
        "Preprocessor",
        "Vectorizer",
        "Base Vectorizer",
        "StopWords",
        "Classifier",
        "Accuracy",
        "Macro F1",
        "Train time (s)",
        "Inference time (s)",
    ]

    save_stage_csv(
        df=results_df,
        columns=csv_columns,
        path="results_DBpedia_V1_baseline.csv",
    )

    print()
    print("Saved: results_DBpedia_V1_baseline.csv")


if __name__ == "__main__":
    main()
