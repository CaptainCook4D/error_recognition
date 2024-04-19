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
)


def get_npz_files(root_dir):
    npz_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, filename))

    return npz_files


def evaluate_func(auc_roc_curve, method_name, result_inner_join, each_error_type):
    true_labels = result_inner_join[each_error_type].values
    predictions = result_inner_join["Predictions"].values
    probabilites = result_inner_join["Probabilities"].values
    true_labels = true_labels.astype(int)
    true_labels = true_labels.reshape(-1)
    predictions = predictions.reshape(-1)
    probabilites = probabilites.reshape(-1)
    thresholded_predictions = np.where(probabilites > 0.5, 1, 0)
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, thresholded_predictions)
    precision = precision_score(true_labels, thresholded_predictions)
    recall = recall_score(true_labels, thresholded_predictions)
    f1 = f1_score(true_labels, thresholded_predictions)
    # Calculate roc auc score
    roc_auc = roc_auc_score(true_labels, probabilites)
    # fpr, tpr, _ = roc_curve(true_labels, probabilites)
    # auc_roc_curve["fpr"][f"{method_name}"] = fpr
    # auc_roc_curve["tpr"][f"{method_name}"] = tpr
    # auc_roc_curve["roc_auc"][f"{method_name}"] = roc_auc

    # Save the plot as a PDF file

    # Show the plot
    plt.show()
    precision *= 100
    recall *= 100
    f1 *= 100

    accuracy = round(accuracy * 100, 2)
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1 = round(f1, 2)
    roc_auc = round(roc_auc, 2)
    return accuracy, precision, recall, f1, roc_auc


plots_path = "plots"
os.makedirs(plots_path, exist_ok=True)
# path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/trained_model_outputs/checkpoints/final"

# New Dataset - New Labels
# Epochs 20
path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/trained_model_outputs/checkpoints_only_main_errors/new_labels/epoch_50"

all_files = get_npz_files(path)
test_labels = pd.read_csv(
    "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_labels_updated/segments/test.csv"
)
print(test_labels.head())

error_columns = [
    "Technique Error",
    "Preparation Error",
    "Measurement Error",
    # "Order Error",
    # "Temperature Error",
    # "Timing Error",
    # "Missing Step",
    # "Other",
]  # Add all error columns here

needed_labels = test_labels[["recording_id"] + error_columns]
# error_types = needed_labels["error_tag"].unique()

all_rows = [
    ["Error", "Method Name", "Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"]
]

for each_error_type in error_columns:
    if each_error_type is None:
        continue
    auc_roc_curve = {"fpr": {}, "tpr": {}, "roc_auc": {}}
    for file in all_files:
        method_name = file.split("/")[-2]
        this_row = [each_error_type, method_name]
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
        mode_per_recording_id = df.groupby("All Recording IDs")["Predictions"].apply(
            lambda x: x.mode().iloc[0]
        )
        mean_per_recording_id = df.groupby("All Recording IDs")["Probabilities"].apply(
            lambda x: x.mean()
        )
        true_labels_per_id = df.groupby("All Recording IDs")["Labels"].unique()
        result_inner_join = true_labels_per_id.to_frame().join(
            mode_per_recording_id.to_frame(), how="inner"
        )
        result_inner_join = result_inner_join.join(
            mean_per_recording_id.to_frame(), how="inner"
        )
        df_for_type_of_errors = result_inner_join.join(
            needed_labels.set_index("recording_id"), how="inner"
        )
        # selected_rows = df_for_type_of_errors[
        #     df_for_type_of_errors["error_tag"] == each_error_type
        # ]
        # We can do this for all videos
        # selected_rows = df_for_type_of_errors[
        #     (df_for_type_of_errors[each_error_type] == 1)
        #     | (df_for_type_of_errors["Labels"] == 0)
        # ]
        selected_rows = df_for_type_of_errors
        print(
            f"Number of unique recording ids: {len(np.unique(outputs['all_recording_ids']))}"
        )
        print(f"Number of recording ids after inner join: {len(selected_rows)}")
        accuracy, precision, recall, f1, roc_auc = evaluate_func(
            auc_roc_curve, method_name, selected_rows, each_error_type
        )

        this_row.extend([accuracy, precision, recall, f1, roc_auc])
        all_rows.append(this_row)

csv_file_path = "error_metrics.csv"

with open(csv_file_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(all_rows)

    # Assuming auc_roc_curve is your dictionary
    # methods = list(auc_roc_curve["fpr"].keys())

    # # Create a new figure to display the ROC AUC curves
    # plt.figure(figsize=(8, 6))

    # Iterate through different methods for comparative analysis
    # for method in methods:
    #     if method == "3dresnet":
    #         continue
    #     fpr = auc_roc_curve["fpr"][method]
    #     tpr = auc_roc_curve["tpr"][method]
    #     roc_auc = auc_roc_curve["roc_auc"][method]
    #     # Plot the ROC curve for the current method
    #     plt.plot(fpr, tpr, label=f"{method} (AUC = {roc_auc:.2f})")

    # # Add a reference diagonal line representing random performance
    # plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    # # Customize plot aesthetics
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Comparative Analysis of ROC AUC Curves")
    # plt.legend(loc="lower right")

    # # Save the generated plot as a PDF file for reference
    # plt.savefig(f"{plots_path}/roc_auc_curve.pdf")

    # # Display the plot
    # plt.show()
