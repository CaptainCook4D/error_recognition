import csv
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import seaborn as sns


def get_npz_files(root_dir):
    npz_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, filename))

    return npz_files


def evaluate_func(auc_roc_curve, method_name, result_inner_join):
    true_labels = result_inner_join["Labels"].values
    predictions = result_inner_join["Predictions"].values
    probabilites = result_inner_join["Probabilities"].values
    true_labels = true_labels.astype(int)
    true_labels = true_labels.reshape(-1)
    predictions = predictions.reshape(-1)
    probabilites = probabilites.reshape(-1)
    thresholded_predictions = np.where(probabilites > 0.5, 1, 0)
    # Calculate evaluation metrics
    precision = precision_score(true_labels, thresholded_predictions)
    recall = recall_score(true_labels, thresholded_predictions)
    f1 = f1_score(true_labels, thresholded_predictions)
    accuracy = accuracy_score(true_labels, thresholded_predictions)

    # Calculate roc auc score
    roc_auc = roc_auc_score(true_labels, probabilites)
    fpr, tpr, _ = roc_curve(true_labels, probabilites)
    auc_roc_curve["fpr"][f"{method_name}"] = fpr
    auc_roc_curve["tpr"][f"{method_name}"] = tpr
    auc_roc_curve["roc_auc"][f"{method_name}"] = roc_auc
    # Show the plot
    plt.show()
    accuracy = round(accuracy * 100, 2)
    precision *= 100
    recall *= 100
    f1 *= 100
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1 = round(f1, 2)
    roc_auc = round(roc_auc, 2)
    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_labels, thresholded_predictions)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix


def compute_mean_first_half(x):
    first_half = x.sort_values().iloc[: len(x) // 2]
    mean = first_half.mean()
    if np.isnan(mean):
        return 0.5
    else:
        return mean


def compute_mode_first_half(x):
    first_half = x.sort_values().iloc[: len(x) // 2]
    return first_half.mode().iloc[0] if not first_half.mode().empty else 0


tasks = ["recognition", "early_prediction"]
for task in tasks:
    plots_path = f"evaluation/plots/{task}"
    os.makedirs("evaluation/plots", exist_ok=True)
    confusion_matrix_path = f"evaluation/confusion_matrix/{task}"
    os.makedirs("evaluation/confusion_matrix", exist_ok=True)
    all_rows = [
        ["Method Name", "Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"]
    ]
    auc_roc_curve = {"fpr": {}, "tpr": {}, "roc_auc": {}}
    # path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/trained_model_outputs/checkpoints/final"

    # New Dataset - New Labels
    # Epochs 20
    path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/trained_model_outputs/checkpoints/new_dataset/epoch_50"
    all_files = get_npz_files(path)
    test_labels = pd.read_csv(
        "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_labels_updated/segments/test.csv"
    )
    needed_labels = test_labels[["recording_id"]]
    print(test_labels.head())

    for file in all_files:
        method_name = file.split("/")[-2]
        this_row = [method_name]
        outputs = np.load(file)
        # Create a DataFrame
        data = {
            "Predictions": outputs["predictions"].reshape(-1),
            "Labels": outputs["true_labels"].reshape(-1),
            "All Indices for Split": outputs["all_indices_for_split"],
            "All Recording IDs": outputs["all_recording_ids"],
            "Probabilities": outputs["probabilities"].reshape(-1),
        }

        df = pd.DataFrame(data)
        true_labels_per_id = df.groupby("All Recording IDs")["Labels"].unique()

        if task == "early_prediction":
            df_grouped = df.groupby("All Recording IDs")
            prob_mean_first_half_per_recording_id = df.groupby("All Recording IDs")[
                "Probabilities"
            ].apply(compute_mean_first_half)
            prob_mean_per_recording_id = prob_mean_first_half_per_recording_id
            pred_mode_first_half_per_recording_id = df.groupby("All Recording IDs")[
                "Predictions"
            ].apply(compute_mode_first_half)

            prediction_mode_per_recording_id = pred_mode_first_half_per_recording_id
        elif task == "recognition":
            prediction_mode_per_recording_id = df.groupby("All Recording IDs")[
                "Predictions"
            ].apply(lambda x: x.mode().iloc[0])
            prob_mean_per_recording_id = df.groupby("All Recording IDs")[
                "Probabilities"
            ].apply(lambda x: x.mean())
        result_inner_join = true_labels_per_id.to_frame().join(
            prediction_mode_per_recording_id.to_frame(), how="inner"
        )
        result_inner_join = result_inner_join.join(
            prob_mean_per_recording_id.to_frame(), how="inner"
        )
        df_for_type_of_errors = result_inner_join.join(
            needed_labels.set_index("recording_id"), how="inner"
        )

        print(
            f"Number of unique recording ids: {len(np.unique(outputs['all_recording_ids']))}"
        )
        print(f"Number of recording ids after inner join: {len(result_inner_join)}")

        accuracy, precision, recall, f1, roc_auc, conf_matrix = evaluate_func(
            auc_roc_curve, method_name, result_inner_join
        )
        # Set up plot aesthetics
        plt.figure(figsize=(6, 6))
        sns.set(font_scale=1.8)
        sns.heatmap(
            conf_matrix,
            annot=True,
            cmap="Blues",
            fmt="d",
            cbar=False,
            square=True,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        # plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

        # Save the plot as a high-quality image (e.g., PDF)
        plt.savefig(
            f"{confusion_matrix_path}{method_name}.pdf",
            bbox_inches="tight",
            format="pdf",
        )

        this_row.extend([accuracy, precision, recall, f1, roc_auc])
        all_rows.append(this_row)

    csv_file_path = f"evaluation/metrics/{task}.csv"
    os.makedirs("evaluation/metrics", exist_ok=True)

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_rows)

    # Assuming auc_roc_curve is your dictionary
    methods = list(auc_roc_curve["fpr"].keys())

    # Create a new figure to display the ROC AUC curves
    plt.figure(figsize=(8, 6))

    # Iterate through different methods for comparative analysis
    for method in methods:
        fpr = auc_roc_curve["fpr"][method]
        tpr = auc_roc_curve["tpr"][method]
        roc_auc = auc_roc_curve["roc_auc"][method]
        # Plot the ROC curve for the current method
        plt.plot(fpr, tpr, label=f"{method} (AUC = {roc_auc:.2f})", linewidth=2)

    # Add a reference diagonal line representing random performance
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    # plt.title(f"ROC AUC Curve ", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(f"{plots_path}roc_auc_curve.pdf", bbox_inches="tight", format="pdf")
    plt.show()

    # Display the plot
    plt.show()
