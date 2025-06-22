SemiAuto-Regression
SemiAuto-Regression is a Python-based microservice that provides a semi-automated pipeline for regression modeling. It orchestrates the end-to-end ML lifecycle—from data ingestion and preprocessing through model training, evaluation, and report generation—while allowing user interaction at key decision points. In practice, the user provides input (e.g. choosing models or parameters) and the service runs the heavy lifting of model building (hyperparameter search, ensembling, etc.) via RESTful APIs. The system packages results (leaderboards, plots, reports) into a user-friendly interface. Key features include:
Regression-focused AutoML: Automates training of multiple regression models (e.g. GBMs, CatBoost) and ensemble methods in a single workflow. Users get a ranked “leaderboard” of models, mirroring the approach of H2O AutoML
automl.org
, but with manual oversight.
Configurable Data Pipeline: Implements a standard ML directory structure (with raw, interim, cleaned, processed data) for reproducibility
github.com
. Sample data (e.g. the Boston housing dataset) is provided to illustrate usage, and data paths can be customized.
REST API & UI: Exposes FastAPI endpoints for triggering tasks (training, inference, reporting) and serves a Jinja2-based web UI for monitoring progress and viewing results. Static assets (HTML/CSS/JS) are served under /regression-static or /static (as configured in app.py
github.com
).
Report Generation: After model runs, the system assembles performance metrics, parameter grids, and visualizations into markdown and PDF reports under a reports/ folder.
Containerized Deployment: A Dockerfile (Python 3.10-slim) is provided for easy deployment. The container exposes port 8030 and launches the FastAPI app with Uvicorn
github.com
.
Architecture Summary
The repository is organized into modular components to separate concerns:
API Server (app.py) – The FastAPI application sets up HTTP routes, CORS middleware, and mounts static/template directories
github.com
. The main entrypoint launches Uvicorn (port 8030 by default) and ensures required folders exist on startup
github.com
github.com
.
Core Library (semiauto_regression/) – Contains Python modules for data loading, preprocessing, model fitting, evaluation, and report synthesis. Models are trained (e.g. CatBoost or others), and results (models, best parameters) are serialized into model/ and reports/. The catboost_info/ folder indicates CatBoost training details (e.g. learning curves) are collected.
Configuration (config/ and intel.yaml) – Holds settings and metadata for experiments. Users can adjust YAML or JSON config to define model hyperparameters, data paths, or pipeline options.
Data Directories (data/) – Follows a layered scheme: raw/ contains original input data, interim/ holds cleaned or preprocessed versions, cleaned/ and processed/ contain training-ready files
github.com
. A feature_store.yaml example is provided under references/ to illustrate feature definitions.
Reports & Notebooks (reports/, notebooks/) – Auto-generated outputs (plots, metrics, project flow PDF) are stored in reports/. Example Jupyter notebooks demonstrate how to run or extend the pipeline.
Docker & Dev Ops – The Dockerfile installs system dependencies (e.g. Git, libgomp1 for CatBoost) and pip packages from requirements-docker.txt
github.com
. Running the container (which exposes port 8030) will start the service so engineers can integrate it into CI/CD or Kubernetes environments.
Overall, SemiAuto-Regression implements a semi-automated pipeline: it automates tasks like hyperparameter search and ensemble training (inspired by AutoML techniques
automl.org
) while the user remains in control of steps like final model selection. This approach reflects how AutoML frameworks aim to simplify modeling so practitioners can focus on preprocessing and deployment
automl.org
.
Installation Instructions
Prerequisites: Python 3.10+ (the Docker image uses Python 3.10-slim) and common data science libraries. You will also need git, Docker (optional), and OS packages for compilation (e.g. libgomp1 for CatBoost).
Clone the repository (or download the latest release):
git clone https://github.com/Akshat-Sharma-110011/SemiAuto-Regression.git
cd SemiAuto-Regression
Install Python dependencies. It is recommended to use a virtual environment.
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
The requirements.txt includes FastAPI, Uvicorn, pandas, scikit-learn, CatBoost, and other ML/utilities.
(Optional) Use Docker. Build and run the Docker image for a consistent environment:
docker build -t semiauto-regression .
docker run --rm -p 8030:8030 semiauto-regression
Inside the container, Uvicorn will start automatically (port 8030)
github.com
. System requirements like libgomp1 are already installed in the Dockerfile.
Verify installation. Once running, point your browser to http://localhost:8030 to access the web UI, or use curl to ping the API.
If installing from source without Docker, ensure the intel.yaml and data/ directories are present (they are part of the repo). Some dependencies like CatBoost or H2O may require extra OS packages (see each library’s documentation).
Usage Guide
Once the service is running (e.g. via Docker or uvicorn app:app), you can interact with it as follows:
Web Dashboard: Navigate to http://<host>:8030/ to view the front-end interface. This UI (built with Jinja2 templates) will show project status, metrics, and links to reports.
API Endpoints: The FastAPI server exposes REST endpoints (see api.txt or inspect app.py). Example usage patterns might include:
POST /train – Trigger the training pipeline. Provide JSON payload (or upload files) specifying the dataset and model settings.
GET /status – Check current job status.
GET /predict – Obtain model predictions for new data.
GET /report – Download the final report (PDF/Markdown) after training.
(Exact routes and payload formats can be found in the code or by browsing http://<host>:8030/docs which serves the Swagger UI automatically.)
Inputs/Outputs: By default, the project uses the example Boston housing dataset (data/raw/data_boston/original.csv). During a run, intermediate files are generated under data/cleaned and data/processed. Final trained models and optimized parameters are written to model/. The pipeline culminates in a reports/ directory containing plots, metrics YAML, and a comprehensive projectflow_report_boston.pdf.
Integration Patterns: SemiAuto-Regression is designed to fit into an API-driven workflow. You might call it from another service or orchestration tool (e.g. via HTTP requests from a CI pipeline). Since all steps are exposed as functions or endpoints, it can be integrated with MLOps tools (Airflow, Kubeflow) or invoked manually as needed.
# Example: run training via HTTP (replace with actual API paths)
curl -X POST http://localhost:8030/train \
     -H "Content-Type: application/json" \
     -d '{"dataset": "data_boston", "model": "catboost", "epochs": 50}'
# Example: start the FastAPI server directly
uvicorn app:app --host 0.0.0.0 --port 8030 --reload
Logs and outputs will appear in the console and in the reports/ folder. Consult the generated markdown reports in reports/readme/ for step-by-step details of the pipeline once complete.
Customization and Configuration
Users can tune or extend the system in several ways:
Pipeline Configurations: Modify config/ files or the intel.yaml to change data paths, feature processing, or hyperparameter grids. For instance, you could adjust which features to include, or define a custom parameter sweep for CatBoost.
Add Models: The core library is modular. To support new regressors, add a model class (following the existing pattern) and update the training logic. The leaderboard ranking and report generation are generic, so new models plug in seamlessly.
Data and Features: Replace or augment the example data under data/. Maintain the same raw/interim/processed structure, or adjust the code to handle a different schema. The sample references/feature_store_boston/feature_store.yaml shows how to document features; you can create similar YAML for your domain.
UI and Reporting: The HTML templates (in templates/) and static assets can be customized for branding or additional visualizations. Reports are generated from these templates via Jinja2. You may add new charts or tables by editing the template files or the report-generation code.
Deployment Settings: Environment variables and Docker settings can be changed (e.g. adjust the exposed port, or disable --reload in production). The Makefile and tox.ini include commands for testing and packaging.
In summary, the architecture is intentionally decoupled: data, models, and orchestration logic live in separate modules. This makes it straightforward for an engineer to tweak one part (like trying a different model library, or connecting a database) without rewriting the whole service.
Limitations and Roadmap
Scope: Currently focused on supervised regression tasks. Classification is not built-in (no objective for multi-class targets yet). Only the Boston dataset is pre-configured; applying new datasets may require code adjustments.
Automation Level: The workflow is semi-automated by design – there are still manual decision points. For example, the user might manually select the “best” model from the leaderboard. Fully hands-off AutoML (no human in the loop) is beyond the current scope.
Feature Engineering: The system does basic preprocessing, but does not perform advanced feature engineering automatically. Users must supply cleaned features or edit the code for new transformations.
Model Diversity: By default it uses a limited set of algorithms (e.g. CatBoost, linear models). The roadmap could include integrating more frameworks (e.g. H2O’s AutoML as a backend) to broaden choices.
Scalability: Designed for moderate datasets; extremely large-scale or real-time streaming scenarios are not yet supported. Enhancements like distributed training or cloud integration (e.g. AWS SageMaker pipelines) are future possibilities.
Testing and Hardening: While basic tests are included (tests/), extensive unit/robustness testing is still in progress. Contributions to improve CI, exception handling, and security would be valuable.
Planned extensions may include support for classification tasks, automated hyperparameter optimization frameworks (Bayesian search, etc.), and richer visualization dashboards. Feedback from users could also drive integration with orchestrators like Airflow, or export to model registries.
Credits
SemiAuto-Regression draws inspiration from the broader AutoML and MLOps research community. For example, H2O AutoML (an open-source, scalable AutoML framework) pioneered the idea of training many models (GBMs, RFs, DNNs, GLMs) and using stacked ensembles to improve accuracy
automl.org
. Its “leaderboard” concept, which ranks models by metrics, is mirrored here. In general, AutoML tools aim to simplify model training via single-line commands, freeing data scientists to focus on data preparation and deployment
automl.org
. Readers may refer to the H2O AutoML paper for more on these principles
automl.org
automl.org
. Other relevant works include surveys of AutoML systems (Auto-WEKA, auto-sklearn, TPOT, etc.), which emphasize end-to-end automation for tabular data. The project also follows software packaging best practices (using a pyproject.toml/setup.py, Docker container, etc.) and is licensed under MIT. References: The H2O AutoML paper (AutoML 2020) is a useful citation for automated regression pipelines
automl.org
automl.org
. It shows how combining random search and model stacking can yield high-quality models with minimal code. (Please cite it if referring to the automated model selection approach used here.)
