# About The Disaster Response Pipeline Project
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

A data set used to create a machine learning pipeline contains real messages that were sent during disaster events. The machine learning pipeline analyzes and categorizes these events, so that an emergency worker can input a new message and get classification results in several categories. 

TODO: The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

# Project Components
- 'app'
    - 'template'
        - 'master.html' 
        - 'go.html' 
    - 'run.py'  
    - 'utils.py' 

- 'data'
  - `disaster_categories.csv`+
  - `disaster_messages.csv`´
  - `process_data.py` (ETL Pipeline: write a data cleaning pipeline that, loads the messages and categories datasets, merges the two datasets, cleans the data, stores it in a SQLite database)
  - `DisasterResponse.db` 

- `models`
  - `train_classifier.py`
  - `utils.py`
  - `train_classifier.py` ML Pipeline: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file
  - `classifier.pkl`

- `README.md`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
