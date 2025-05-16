import subprocess
import sys

def run_pipeline():
    # Define the pipeline steps in order
    pipeline_steps = [
        "python src/data/data_ingestion.py",
        "python src/data/data_preprocessing.py",
        "python src/features/feature_engineering.py",
        "python src/model/model_building.py",
        "python src/model/model_evaluation.py",
        "python src/model/model_optimization.py",
        "python src/model/model_evaluation.py",  # Re-evaluate after optimization
        "python src/visualization/projectflow_report.py"
    ]

    # Execute each step sequentially
    for step in pipeline_steps:
        try:
            print(f"\n\033[1m=== RUNNING STEP: {step} ===\033[0m")
            subprocess.run(step, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"\n\033[31mERROR in step '{step}': {e}\033[0m")
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
    print("\n\033[1;32m=== PIPELINE EXECUTED SUCCESSFULLY ===\033[0m")