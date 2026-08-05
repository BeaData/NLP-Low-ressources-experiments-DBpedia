#!/usr/bin/python3

from utils import load, prepare
import pandas as pd


def create_nested_datasets(
    df: pd.DataFrame,
    output_prefix: str,
    sizes: list[int],
    class_column: str
) -> None:
    """
    Create nested and approximately balanced datasets.

    Each class is shuffled once using a fixed random seed.
    Larger datasets contain all observations from smaller
    datasets.

    Example:
        50 ⊂ 200 ⊂ 500 ⊂ 2000 ⊂ 10000

    When a requested size is not divisible by the number
    of classes, the remaining observations are distributed
    across the first classes. The difference between class
    sizes is therefore at most one observation.
    """

    groups = {}

    # Shuffle each class once
    for label in sorted(df[class_column].unique()):

        groups[label] = (
            df[df[class_column] == label]
            .sample(
                frac=1,
                random_state=42
            )
            .reset_index(drop=True)
        )

    n_classes = len(groups)

    print(f"Number of classes: {n_classes}")
    print()

    # Check that every class contains enough observations
    largest_size = max(sizes)

    base = largest_size // n_classes
    remainder = largest_size % n_classes

    maximum_per_class = (
        base + 1
        if remainder > 0
        else base
    )

    for label, group in groups.items():

        if len(group) < maximum_per_class:

            raise ValueError(
                f"Class {label} contains only "
                f"{len(group)} observations, but "
                f"{maximum_per_class} are required."
            )

    for size in sizes:

        base = size // n_classes
        remainder = size % n_classes

        parts = []

        for i, label in enumerate(sorted(groups)):

            n_samples = base

            if i < remainder:
                n_samples += 1

            parts.append(
                groups[label].iloc[:n_samples]
            )

        subset = pd.concat(
            parts,
            ignore_index=True
        )

        # Shuffle the final dataset so that classes are mixed
        subset = (
            subset
            .sample(
                frac=1,
                random_state=42
            )
            .reset_index(drop=True)
        )

        filename = f"{output_prefix}_{size}.csv"

        subset.to_csv(
            filename,
            index=False
        )

        class_counts = (
            subset[class_column]
            .value_counts()
            .sort_index()
        )

        print(
            f"{filename:<28} "
            f"{len(subset):>6} articles"
        )

        print(
            f"Class distribution: "
            f"{class_counts.to_dict()}"
        )

        print()


def create_balanced_test_dataset(
    df: pd.DataFrame,
    output_filename: str,
    class_column: str,
    samples_per_class: int
) -> None:
    """
    Create a balanced test dataset.

    The same number of observations is sampled from
    every class.
    """

    parts = []

    for label in sorted(df[class_column].unique()):

        class_data = df[
            df[class_column] == label
        ]

        if len(class_data) < samples_per_class:

            raise ValueError(
                f"Class {label} contains only "
                f"{len(class_data)} observations, but "
                f"{samples_per_class} are required."
            )

        sampled_class = class_data.sample(
            n=samples_per_class,
            random_state=42
        )

        parts.append(sampled_class)

    test_subset = pd.concat(
        parts,
        ignore_index=True
    )

    # Shuffle the final test dataset
    test_subset = (
        test_subset
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(drop=True)
    )

    text, labels = prepare(test_subset)

    df_test = pd.DataFrame(
        {
            "Text": text,
            "Class Index": labels
        }
    )

    df_test.to_csv(
        output_filename,
        index=False
    )

    class_counts = (
        df_test["Class Index"]
        .value_counts()
        .sort_index()
    )

    print(
        f"{output_filename:<28} "
        f"{len(df_test):>6} articles"
    )

    print(
        f"Class distribution: "
        f"{class_counts.to_dict()}"
    )


def main():
    """
    Create nested DBpedia training datasets and a
    balanced DBpedia test dataset.

    ```
    Training sizes:
        50, 200, 500, 2000, and 10000

    Test size:
        14000 observations
        1000 observations per class
    """

    train = load("train.csv")

    if train is None:
        return

    train_sizes = [
        50,
        200,
        500,
        2000,
        10000
    ]

    print(
        "Creating nested DBpedia "
        "training datasets"
    )

    print()

    create_nested_datasets(
        df=train,
        output_prefix="dbpedia_train",
        sizes=train_sizes,
        class_column="label"
    )

    print(
        "Creating balanced DBpedia "
        "test dataset"
    )

    print()

    test = load("test.csv")

    if test is None:
        return
    print(test.columns.tolist())

    print(
        "Duplicate columns:",
        test.columns[
            test.columns.duplicated()
        ].tolist()
    )

    create_balanced_test_dataset(
        df=test,
        output_filename="dbpedia_test_14000.csv",
        class_column="label",
        samples_per_class=1000
    )


if __name__ == "__main__":
    main()
