from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from io import BytesIO
import pandas as pd
import os
import yaml
import tempfile
from pathlib import Path
import uvicorn
import shutil
import sys
# Import internal modules
from src.data.data_ingestion import create_data_ingestion
from utils import (load_yaml, update_intel_yaml)
from src.data.data_cleaning import main as data_cleaning_main
from src.features.feature_engineering import run_feature_engineering
from src.model.model_building import ModelBuilder
from src.model.model_evaluation import run_evaluation, get_evaluation_summary
from src.model.model_optimization import optimize_model

app = FastAPI(title="SemiAuto Regression", version="1.0")

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

class PreprocessingConfig(BaseModel):
    missing_values: Optional[str] = None
    handle_duplicates: bool = True
    outliers: Optional[str] = None
    skewedness: Optional[str] = None
    scaling: Optional[str] = None
    encoding: Optional[str] = None
    drop_first: Optional[bool] = False


class FeatureEngineeringRequest(BaseModel):
    use_feature_tools: bool = False
    use_shap: bool = False
    n_features: int = 20


class ModelBuildRequest(BaseModel):
    model_name: str
    custom_params: Optional[Dict[str, Any]] = None


class OptimizationRequest(BaseModel):
    optimize: bool = True
    method: str = "1"  # "1"=GridSearch, "2"=Optuna
    n_trials: int = 50
    metric: str = "1"  # e.g. "1"=RMSE


# Route to main page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to data upload page
@app.get("/data-upload", response_class=HTMLResponse)
async def data_upload_page(request: Request):
    return templates.TemplateResponse("data_upload.html", {"request": request})


# Route to preprocessing page
@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_page(request: Request):
    # Try to load intel.yaml to check if data has been uploaded
    try:
        intel = load_yaml("intel.yaml")
        # Check if we have columns information to display
        feature_store = {}
        if 'feature_store_path' in intel and os.path.exists(intel['feature_store_path']):
            feature_store = load_yaml(intel['feature_store_path'])

        return templates.TemplateResponse("preprocessing.html", {
            "request": request,
            "dataset_name": intel.get('dataset_name', ''),
            "target_column": intel.get('target_column', ''),
            "numerical_cols": feature_store.get('numerical_cols', []),
            "categorical_cols": feature_store.get('categorical_cols', []),
            "nulls": feature_store.get('contains_null', []),
            "outliers": feature_store.get('contains_outliers', []),
            "skewed": feature_store.get('skewed_cols', [])
        })
    except:
        # If not, redirect to upload page
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please upload data first",
            "redirect_url": "/data-upload"
        })


# Route to feature engineering page
@app.get("/feature-engineering", response_class=HTMLResponse)
async def feature_engineering_page(request: Request):
    try:
        intel = load_yaml("intel.yaml")
        if not intel.get('train_preprocessed_path'):
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete preprocessing first",
                "redirect_url": "/preprocessing"
            })
        return templates.TemplateResponse("feature_engineering.html", {"request": request})
    except:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please upload data and complete preprocessing first",
            "redirect_url": "/data-upload"
        })


# Route to model building page
@app.get("/model-building", response_class=HTMLResponse)
async def model_building_page(request: Request):
    try:
        intel = load_yaml("intel.yaml")
        # Check for the correct key set during feature engineering
        if not intel.get('train_transformed_path'):
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please complete feature engineering first",
                "redirect_url": "/feature-engineering"
            })

        builder = ModelBuilder()
        available_models = builder.get_available_models()

        return templates.TemplateResponse("model_building.html", {
            "request": request,
            "available_models": available_models
        })
    except:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# Route to model optimization page
@app.get("/optimization", response_class=HTMLResponse)
async def optimization_page(request: Request):
    try:
        intel = load_yaml("intel.yaml")
        if not intel.get('model_path'):
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Please build a model first",
                "redirect_url": "/model-building"
            })

        return templates.TemplateResponse("optimization.html", {"request": request})
    except:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# Add these endpoints in app.py after the existing /api/generate-report endpoint

@app.get("/api/download-model")
def download_model():
    intel = load_yaml("intel.yaml")
    dataset_name = intel.get("dataset_name", "unnamed")

    # Check for optimized model first
    model_dir = Path(f"model/model_{dataset_name}")
    optimized_model_path = model_dir / "optimized_model.pkl"
    standard_model_path = model_dir / "model.pkl"

    # Choose optimized model if it exists, otherwise standard model
    if optimized_model_path.exists():
        model_path = str(optimized_model_path)
        filename = f"optimized_model_{dataset_name}.pkl"
    elif standard_model_path.exists():
        model_path = str(standard_model_path)
        filename = f"model_{dataset_name}.pkl"
    else:
        # Fallback to what's in intel.yaml
        model_path = intel.get("model_path")
        filename = os.path.basename(model_path) if model_path else "model.pkl"

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=model_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/api/download-pipeline")
def download_pipeline():
    intel = load_yaml("intel.yaml")
    dataset_name = intel.get("dataset_name", "unnamed")

    # Define paths for the different pipeline types
    pipeline_dir = Path(f"model/pipelines/preprocessing_{dataset_name}")
    preprocessing_path = pipeline_dir / "preprocessing.pkl"

    # Check if the combined processor pipeline exists
    processor_path = Path(f"model/pipelines/performance_{dataset_name}/processor.pkl")
    if processor_path.exists():
        return FileResponse(
            path=str(processor_path),
            filename=f"combined_pipeline_{dataset_name}.pkl",
            media_type="application/octet-stream"
        )

    # Fall back to preprocessing pipeline if combined doesn't exist
    if preprocessing_path.exists():
        return FileResponse(
            path=str(preprocessing_path),
            filename=f"preprocessing_pipeline_{dataset_name}.pkl",
            media_type="application/octet-stream"
        )

    # Last resort: use the path from intel.yaml
    pipeline_path = intel.get("preprocessing_pipeline_path")
    if not pipeline_path or not os.path.exists(pipeline_path):
        raise HTTPException(status_code=404, detail="Pipeline file not found")

    return FileResponse(
        path=pipeline_path,
        filename=os.path.basename(pipeline_path),
        media_type="application/octet-stream"
    )


# Add a new endpoint to download the feature engineering pipeline specifically
@app.get("/api/download-feature-pipeline")
def download_feature_pipeline():
    intel = load_yaml("intel.yaml")
    dataset_name = intel.get("dataset_name", "unnamed")

    # Define path for the feature engineering pipeline
    feature_pipeline_path = Path(f"model/pipelines/performance_{dataset_name}/transformation.pkl")

    if not feature_pipeline_path.exists():
        raise HTTPException(status_code=404, detail="Feature engineering pipeline not found")

    return FileResponse(
        path=str(feature_pipeline_path),
        filename=f"feature_pipeline_{dataset_name}.pkl",
        media_type="application/octet-stream"
    )
# Route to results page
@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    try:
        intel = load_yaml("intel.yaml")
        if not intel.get('performance_metrics_path'):
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "No model evaluation results available",
                "redirect_url": "/model-building"
            })

        # Load evaluation metrics
        metrics = load_yaml(intel['performance_metrics_path'])

        return templates.TemplateResponse("results.html", {
            "request": request,
            "metrics": metrics,
            "dataset_name": intel.get('dataset_name', 'unnamed')
        })
    except:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "Please complete previous steps first",
            "redirect_url": "/data-upload"
        })


# API endpoints
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), target_column: str = Form(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        contents = await file.read()
        temp.write(contents)
        temp.flush()
        path = temp.name

    df = pd.read_csv(path)
    columns = df.columns.tolist()

    ingestion = create_data_ingestion()
    with open(path, "rb") as f:
        ingestion.run_ingestion_pipeline(f, file.filename, target_column)

    os.unlink(path)
    ingestion.save_intel_yaml()

    # Run data cleaning after ingestion
    try:
        data_cleaning_main()  # This processes the raw data and updates intel.yaml
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

    return {
        "message": "Data ingestion and cleaning completed",
        "columns": columns
    }


# Update the /api/preprocess endpoint to use cleaned data paths
@app.post("/api/preprocess")
def preprocess_data(config: PreprocessingConfig):
    from src.data.data_preprocessing import (
        PreprocessingPipeline, PreprocessingParameters,
        check_for_duplicates, get_numerical_columns,
        get_categorical_columns, recommend_skewness_transformer,
        load_yaml, update_intel_yaml
    )

    intel = load_yaml("intel.yaml")
    # Use cleaned data paths from data_cleaning
    train_df = pd.read_csv(intel['cleaned_train_path'])
    if 'cleaned_test_path' in intel:
        test_df = pd.read_csv(intel['cleaned_test_path'])
    else:
        test_df = pd.DataFrame()

    feature_store = load_yaml(intel['feature_store_path'])

    numerical = feature_store.get('numerical_cols', [])
    categorical = feature_store.get('categorical_cols', [])
    nulls = [col for col in feature_store.get('contains_null', []) if col != intel['target_column']]
    outliers = [col for col in feature_store.get('contains_outliers', []) if col != intel['target_column']]
    skewed = [col for col in feature_store.get('skewed_cols', []) if col != intel['target_column']]

    pipeline = PreprocessingPipeline({
        'dataset_name': intel['dataset_name'],
        'target_col': intel['target_column'],
        'feature_store': feature_store
    }, PreprocessingParameters())

    if config.missing_values and nulls:
        pipeline.handle_missing_values(config.missing_values, nulls)
    if config.outliers and outliers:
        pipeline.handle_outliers(config.outliers, outliers)
    if config.skewedness and skewed:
        pipeline.handle_skewed_data(config.skewedness, skewed)
    if config.scaling and numerical:
        pipeline.scale_numerical_features(config.scaling, numerical)
    if config.encoding and categorical:
        pipeline.encode_categorical_features(config.encoding, categorical, config.drop_first)

    pipeline.fit(train_df)
    train_p = pipeline.transform(train_df, handle_duplicates=config.handle_duplicates)
    test_p = pipeline.transform(test_df, handle_duplicates=False)

    interim_dir = Path(f"data/interim/data_{intel['dataset_name']}")
    interim_dir.mkdir(parents=True, exist_ok=True)
    train_path = interim_dir / "train_preprocessed.csv"
    test_path = interim_dir / "test_preprocessed.csv"

    pipeline_dir = Path(f"model/pipelines/preprocessing_{intel['dataset_name']}")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = pipeline_dir / "preprocessing.pkl"

    train_p.to_csv(train_path, index=False)
    test_p.to_csv(test_path, index=False)
    pipeline.save(str(pipeline_path))

    update_intel_yaml("intel.yaml", {
        "train_preprocessed_path": str(train_path),
        "test_preprocessed_path": str(test_path),
        "preprocessing_pipeline_path": str(pipeline_path)
    })

    return {"message": "Preprocessing completed"}


@app.post("/api/feature-engineering")
def feature_engineering(request: FeatureEngineeringRequest):
    result = run_feature_engineering(
        config_path="intel.yaml",
        use_feature_tools=request.use_feature_tools,
        use_shap=request.use_shap,
        n_features=request.n_features
    )
    return result


@app.get("/api/available-models")
def get_model_list():
    builder = ModelBuilder()
    return builder.get_available_models()


@app.post("/api/build-model")
def build_model(request: ModelBuildRequest):
    builder = ModelBuilder()
    result = builder.process_model_request(
        model_name=request.model_name,
        custom_params=request.custom_params
    )
    evaluation = run_evaluation("intel.yaml")
    return {"build_result": result, "evaluation_result": evaluation}


@app.post("/api/optimize")
def optimize(request: OptimizationRequest):
    if not request.optimize:
        return {"message": "Optimization skipped."}

    result = optimize_model(
        optimize=request.optimize,
        method=request.method,
        n_trials=request.n_trials,
        metric=request.metric,
        config_overrides=None
    )

    evaluation = run_evaluation("intel.yaml")
    return {"optimization_result": result, "evaluation_result": evaluation}

@app.get("/api/download-model")
def download_model():
    """
    Endpoint to download the trained model file.
    Returns the optimized model if available, otherwise returns the standard model.
    """
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get('dataset_name', 'unnamed')

        # Check for optimized model first
        optimized_model_path = f"model/model_{dataset_name}/optimized_model.pkl"
        standard_model_path = intel.get("model_path")

        if os.path.exists(optimized_model_path):
            model_path = optimized_model_path
            filename = f"optimized_model_{dataset_name}.pkl"
        elif standard_model_path and os.path.exists(standard_model_path):
            model_path = standard_model_path
            filename = f"model_{dataset_name}.pkl"
        else:
            raise HTTPException(status_code=404, detail="Model file not found")

        return FileResponse(
            path=model_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")


@app.get("/api/download-pipeline")
def download_pipeline():
    """
    Endpoint to download the preprocessing and feature engineering pipeline.
    Returns the combined processor pipeline if available, otherwise returns
    the preprocessing pipeline.
    """
    try:
        intel = load_yaml("intel.yaml")
        dataset_name = intel.get('dataset_name', 'unnamed')

        # Check for combined processor pipeline first
        processor_path = f"model/pipelines/preprocessing_{dataset_name}/processor.pkl"
        preprocessing_path = intel.get("preprocessing_pipeline_path")

        if os.path.exists(processor_path):
            pipeline_path = processor_path
            filename = f"combined_pipeline_{dataset_name}.pkl"
        elif preprocessing_path and os.path.exists(preprocessing_path):
            pipeline_path = preprocessing_path
            filename = f"preprocessing_pipeline_{dataset_name}.pkl"
        else:
            raise HTTPException(status_code=404, detail="Pipeline file not found")

        return FileResponse(
            path=pipeline_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading pipeline: {str(e)}")
@app.get("/api/generate-report")
def generate_report():
    from src.visualization.projectflow_report import ProjectFlowReport
    report_generator = ProjectFlowReport("intel.yaml")
    report_generator.generate_report()
    intel = load_yaml("intel.yaml")
    dataset_name = intel.get("dataset_name", "unnamed")
    report_path = f"reports/pdf/projectflow_report_{dataset_name}.pdf"

    if not os.path.exists(report_path):
        raise HTTPException(status_code=500, detail="Report generation failed")

    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="application/pdf"
    )


if __name__ == "__main__":
    # Create required directories if they don't exist
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)