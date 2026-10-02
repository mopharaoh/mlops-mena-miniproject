import sys

import mlflow


def main() -> None:
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name("house-prices")

    if experiment is None:
        print("MLflow experiment not found.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if len(runs) < 2:
        print("Not enough runs for quality comparison.")
        return

    candidate = runs[0]
    production = runs[1]

    candidate_mae = candidate.data.metrics["mae"]
    production_mae = production.data.metrics["mae"]

    allowed_mae = production_mae * 1.05

    print(f"Production MAE: {production_mae}")
    print(f"Candidate MAE: {candidate_mae}")
    print(f"Allowed MAE: {allowed_mae}")

    if candidate_mae > allowed_mae:
        print("QUALITY GATE FAILED: " "MAE regressed by more than 5%.")
        sys.exit(1)

    print("QUALITY GATE PASSED.")


if __name__ == "__main__":
    main()
