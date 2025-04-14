# navigate to directory mlruns are stored
cd unified_experiment/
# start mlflow server (& tells it to run in the background) 
mlflow server --host 0.0.0.0 --port 8080 &
# navigate to api directory
cd ../api
# run api script
uvicorn api_server:app --host 0.0.0.0 --port 8000