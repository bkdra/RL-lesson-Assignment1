from utils import compare_experiment_pairs

if __name__ == "__main__":
    # Compare the training curves of the two experiments
    compare_experiment_pairs(
            csv_path_pairs=[["outputs/part_b_c_1/training_metrics_depth2.csv", "outputs/part_b_c_2/training_metrics_depth2.csv"], 
                       ["outputs/part_b_c_1/training_metrics_base.csv", "outputs/part_b_c_2/training_metrics_base.csv"], 
                       ["outputs/part_b_c_1/training_metrics_depth4.csv", "outputs/part_b_c_2/training_metrics_depth4.csv"]],
            labels=["Depth 2", "Base (depth 3)", "Depth 4"],
            out_path="outputs/exp/experiments_depth.png", 
            include_epsilon=False,
    )
    # csv_paths: list, labels: list = None, out_path: str = "outputs/part_b_c_1/experiments_comparison.png"