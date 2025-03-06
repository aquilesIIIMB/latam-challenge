# Flight Delay Prediction Model & API Documentation

## 1. Project Overview

This project implements an end-to-end machine learning solution to predict flight delays at Santiago de Chile (SCL) airport. The implementation transforms exploratory data science work into a production-ready API service, capable of handling prediction requests in real-time.

### 1.1 Problem Statement

The challenge is to build a system that predicts whether a flight will be delayed based on several features including flight operator, flight type, and month of operation. A flight is considered delayed if it departs more than 15 minutes after its scheduled departure time.

### 1.2 Solution Architecture

The solution consists of three main components:

1. **Machine Learning Model**: An XGBoost classifier that predicts flight delays based on feature engineering and the top 10 most important features.
2. **REST API**: A FastAPI-based interface that exposes the model to external clients.
3. **CI/CD Pipeline**: GitHub Actions workflows that automate testing, building, and deployment to Google Cloud Run.

### 1.3 Dataset Description

The model was trained on real flight data with the following key attributes:

| Original Column | Description |
|-----------------|-------------|
| `OPERA` | Airline operator name |
| `TIPOVUELO` | Flight type (I=International, N=National) |
| `MES` | Month of operation (1-12) |

### 1.4 Project Structure

```
.
├── .github/workflows      # CI/CD configuration
│   ├── ci.yml             # Continuous Integration workflow
│   └── cd.yml             # Continuous Delivery workflow
├── challenge              # Core application code
│   ├── __init__.py        # Package initialization
│   ├── api.py             # FastAPI implementation
│   ├── exploration.ipynb  # Jupyter Notebooks with the base code of the model
│   └── model.py           # ML model implementation
├── data                   # Data directory
│   └── data.csv           # Dataset to train the model
├── docs                   # Documentation
├── reports                # Test reports and coverage
├── tests                  # Test suites
│   ├── api                # API tests
│   ├── model              # Model tests
│   └── stress             # Load/stress tests
├── Dockerfile             # Container configuration
├── Makefile               # Build and test automation
├── README.md              # Project readme
└── requirements*.txt      # Dependency specifications
```

## 2. Model Implementation

The model implementation involved translating a Jupyter notebook into a well-structured Python class with proper error handling, logging, and testing.

### 2.1 Data Preprocessing

The preprocessing pipeline transforms raw flight data into features suitable for the model:

1. **Feature Engineering**: Creates one-hot encodings for categorical variables (OPERA, TIPOVUELO, MES)
2. **Feature Selection**: Focuses on the top 10 most important features based on feature importance analysis
3. **Target Variable Creation**: Calculates the delay status by comparing scheduled and actual departure times

### 2.2 Model Selection and Training

After evaluating multiple models in the exploration phase, **XGBoost with class balancing** was selected as the best performing model. Key advantages:

- Better handling of imbalanced classes (much fewer delayed flights than on-time)
- Good performance with feature importance-based selection
- Efficient training and prediction times
- Good recall for the positive class (delayed flights)

### 2.3 Model Code Structure

The `DelayModel` class in `model.py` includes:

- `preprocess()`: Prepares data for training or prediction
- `fit()`: Trains the model with class balancing
- `predict()`: Makes predictions on new data
- Helper methods for model persistence and data transformation

### 2.4 Model Testing

Model testing was performed using the pytest framework to ensure:

- Correct preprocessing for both training and inference
- Model performance metrics meet requirements
- Proper handling of edge cases and errors

To run the model tests:

```bash
make model-test
```

Test results show that the model achieves:
- Class 0 (on-time): Precision ≈ 0.94, Recall ≈ 0.58, F1 ≈ 0.72
- Class 1 (delayed): Precision ≈ 0.11, Recall ≈ 0.61, F1 ≈ 0.19

## 3. API Implementation

The API exposes the model through a RESTful interface built with FastAPI.

### 3.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for the API |
| `/predict` | POST | Prediction endpoint that accepts flight data and returns delay predictions |

### 3.2 Input and Output Schemas

**Input Schema** (example):
```json
{
  "flights": [
    {
      "OPERA": "Grupo LATAM",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

**Output Schema**:
```json
{
  "predict": [0]  // 0 = On-time, 1 = Delayed
}
```

### 3.3 API Testing

API tests validate:
- Proper response formatting
- Error handling for invalid inputs
- Consistency with model predictions

To run the API tests:

```bash
make api-test
```

### 3.4 Running the API Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
uvicorn challenge.api:app --reload
```

3. Access the API documentation:
```
http://127.0.0.1:8000/docs
```

4. Test with a sample request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]}'
```

## 4. Containerization with Docker

### 4.1 Dockerfile Analysis

The Dockerfile uses a Python 3.9 slim base image and includes:
- Installation of necessary build tools
- Installation of Python dependencies
- Copy of application code
- Configuration of the entry point command

### 4.2 Building the Docker Image

```bash
docker build -t flight-delay-api:latest .
```

### 4.3 Running the Container Locally

```bash
docker run -p 8080:8080 flight-delay-api:latest
```

Test the containerized API:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]}'
```

## 5. Cloud Deployment

The application is deployed on Google Cloud Run, providing a scalable, serverless environment.

### 5.1 Deployment Architecture

Google Cloud Run was selected for deployment because:
- Serverless, automatic scaling
- Only pay for resources used
- Easy integration with CI/CD pipelines
- Built-in container registry

### 5.2 Accessing the Deployed API

The API is available at: [https://flight-delay-api-1099093996594.us-central1.run.app](https://flight-delay-api-1099093996594.us-central1.run.app)

Sample request to the cloud API:
```bash
curl -X POST https://flight-delay-api-1099093996594.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]}'
```

Sample response:
```json
{"predict":[0]}
```

### 5.3 Stress Testing the Deployed API

The API was stress tested using Locust to ensure it can handle significant load:

```bash
make stress-test
```

Results showed the API can handle up to 100 concurrent users with acceptable response times.

## 6. CI/CD Implementation

### 6.1 Continuous Integration (CI)

The CI pipeline (`ci.yml`) runs on every push and pull request, performing:

1. Code checkout
2. Python environment setup
3. Dependencies installation
4. Model tests execution
5. API tests execution
6. Linting with flake8
7. Building of Python package

This ensures that code quality is maintained and that all tests pass before changes are merged.

### 6.2 Continuous Delivery (CD)

The CD pipeline (`cd.yml`) runs on pushes to the main branch:

1. Builds a Docker image with the latest code
2. Pushes the image to Google Container Registry
3. Deploys the image to Google Cloud Run
4. Performs stress tests on the deployed API

## 7. Challenge Completion Documentation

The challenge has been completed successfully with:

1. Transcription of the Jupyter notebook into a well-structured Python class
2. Development of a FastAPI application exposing the model
3. Deployment to GCP Cloud Run
4. Implementation of CI/CD pipelines

### 7.1 Key Features of the Implementation

- **Code Quality**: Proper error handling, logging, and separation of concerns
- **Performance**: Focus on the top 10 features for optimized prediction
- **Robustness**: Comprehensive test suite for model and API
- **Scalability**: Cloud Run deployment with automatic scaling
- **Maintainability**: CI/CD pipelines for automated testing and deployment

## 8. Future Improvements

Several enhancements could be made to further improve the solution:

1. **Model Performance**: Experiment with more advanced feature engineering and ensemble methods to improve prediction accuracy, especially for the delayed class where precision could be improved.

2. **API Features**: Add batch prediction capabilities for processing multiple flights simultaneously and implement detailed model explanation endpoints that provide insights into prediction factors.

3. **Infrastructure**: Set up comprehensive monitoring and alerting for API metrics including response times, error rates, and resource utilization to ensure optimal performance.

4. **Security**: Implement authentication mechanisms, rate limiting, and input validation to protect the API from potential misuse and ensure data privacy.

5. **Documentation**: Create an interactive API documentation page with example use cases and integration guides for different client applications.

6. **BentoML Integration**: Implement BentoML for model serving, which would provide a standardized way to package, deploy, and monitor machine learning models. BentoML would simplify deployment not just to Cloud Run but across multiple cloud platforms, offering features like model versioning, A/B testing, and automatic scaling with minimal configuration.

7. **Dependency Management**: Replace the current requirements.txt approach with TOML-based dependency management (using tools like Poetry or Pipenv). TOML offers better readability, explicit version constraints, development vs. production dependency separation, and better handling of complex dependency trees, making the project more maintainable long-term.

## 9. Conclusion

This project demonstrates the end-to-end implementation of a machine learning model in a production environment. From data preprocessing to model training and from API development to cloud deployment, each component has been carefully designed, implemented, and tested.

The most challenging aspects were:
- Balancing model performance with inference speed
- Ensuring robust error handling in the API
- Setting up the CI/CD pipeline for automated deployment

The final solution provides a scalable, reliable API for flight delay predictions that can be easily maintained and extended in the future.
