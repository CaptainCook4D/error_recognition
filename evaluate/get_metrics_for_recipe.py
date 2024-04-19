import csv
import json
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
    # Calculate roc auc score
    # roc_auc = roc_auc_score(true_labels, probabilites)
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

    precision = round(precision, 2)
    recall = round(recall, 2)
    f1 = round(f1, 2)
    # roc_auc = round(roc_auc, 2)
    return precision, recall, f1


plots_path = "plots"
metrics_path = "metrics"
os.makedirs(plots_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)
path = "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/trained_model_outputs/checkpoints/2023-08-25-23-19-08"
all_files = get_npz_files(path)
test_labels = pd.read_csv(
    "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_labels_segments/test.csv"
)
print(test_labels.head())

recipes = json.load(
    open(
        "/home/sxa180157/projects/error_dataset/supervised_benchmarks/Supervised-Benchmarks-for-Error-Dataset/sl_methods/src/evaluate/recipe.json"
    )
)
for each_recipe in recipes:
    all_rows = [["Method Name", "Precision", "Recall", "F1 Score"]]
    auc_roc_curve = {"fpr": {}, "tpr": {}, "roc_auc": {}}
    for file in all_files:
        method_name = file.split("/")[-2]
        if method_name in ["slowfast_1024", "x3d"]:
            continue
        elif method_name == "slowfast_last":
            method_name = "Slowfast"
        this_row = [method_name]
        outputs = np.load(file)
        # Create a DataFrame
        data = {
            "Predictions": outputs["predictions"].reshape(-1),
            "Labels": outputs["true_labels"].reshape(-1),
            "All Indices for Split": outputs["all_indices_for_split"],
            "Rec_IDs": outputs["all_recording_ids"],
            "Probabilities": outputs["probabilities"].reshape(-1),
            "Second_ID": outputs["all_recording_ids"],
        }

        df = pd.DataFrame(data)
        recipe_ids = df["Second_ID"].str.split("_").str.get(0)
        df["Recipe Index"] = recipe_ids
        recipe_index = each_recipe
        recipe_name = recipes[recipe_index]
        df = df[df["Recipe Index"] == each_recipe]

        mode_per_recording_id = df.groupby("Rec_IDs")["Predictions"].apply(
            lambda x: x.mode().iloc[0]
        )
        mean_per_recording_id = df.groupby("Rec_IDs")["Probabilities"].apply(
            lambda x: x.mean()
        )
        true_labels_per_id = df.groupby("Rec_IDs")["Labels"].unique()
        result_inner_join = true_labels_per_id.to_frame().join(
            mode_per_recording_id.to_frame(), how="inner"
        )
        result_inner_join = result_inner_join.join(
            mean_per_recording_id.to_frame(), how="inner"
        )

        print(
            f"Number of unique recording ids: {len(np.unique(outputs['all_recording_ids']))}"
        )
        print(f"Number of recording ids after inner join: {len(result_inner_join)}")
        precision, recall, f1 = evaluate_func(
            auc_roc_curve, method_name, result_inner_join
        )

        this_row.extend([precision, recall, f1])
        all_rows.append(this_row)

    csv_file_path = f"{metrics_path}/{recipe_name}_metrics.csv"

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_rows)

    # Assuming auc_roc_curve is your dictionary
    methods = list(auc_roc_curve["fpr"].keys())

    # Create a new figure to display the ROC AUC curves
    plt.figure(figsize=(8, 6))

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
