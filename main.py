from src.data_preprocessing import preprocess_run
from src.feature_engineering import add_features, prepare_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.utils.io_utils import load_csv, save_csv
from src.run_inference_future_predict import run_inference_future_predict
from src.config.config import (
    MODEL_PATH,
    INFERENCE_INPUT_PATH,
    INFERENCE_OUTPUT_PATH,
)

#adding loggers

from src.utils.logger import get_logger
logger = get_logger("RetailForecastPipeline")

def main():
    #print("🚀 Starting Retail Demand Forecasting Pipeline\n")
    logger.info("Starting Retail Demand Forecasting Pipeline")

    # Step 1: Preprocess data
    df = preprocess_run()  # MUST return a DataFrame
    #print("Data preprocessing completed")
    logger.info("Data preprocessing completed")

    # Step 2: Feature engineering
    df = add_features(df)  # MUST return df with new features
    #print("Feature engineering completed")
    logger.info("Feature engineering completed")
    logger.debug(f"Columns after feature engineering: {df.columns.tolist()}")

    # Step 3: Prepare features for modeling
    X_train, X_test, y_train, y_test = prepare_features(df)
    print("Features prepared & train/test split done")
    logger.info("Features prepared & train/test split done")

    # # Step 4: Train model using train split
    # model = train_model(X_train, y_train, model_path="models/random_forest.pkl")
    # #print("Model training & saving completed")
    # logger.info("Model training & saving completed")

    # Step 4: Train model using train split
    model = train_model(X_train, y_train, model_path=MODEL_PATH)
    logger.info(f"Model training & saving completed at {MODEL_PATH}")

    # Step 5: Evaluate model using test split
    evaluate_model(model, X_test, y_test)
    #print("Model evaluation completed")
    logger.info("Model evaluation completed")

    # step 6: Run Inference (NEW)
    run_inference_future_predict(
        model_path=MODEL_PATH,
        input_path=INFERENCE_INPUT_PATH,
        output_path=INFERENCE_OUTPUT_PATH
    )
    logger.info("Inference completed & predictions saved")

    #print("\n Pipeline completed successfully")
    logger.info("\n Pipeline completed successfully")

if __name__ == "__main__":
    main()