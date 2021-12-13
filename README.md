# Bot Detection

A marketplace is being attacked by bots that produce fake clicks and leads. Themarketplace reputation might be affected if sellers get tons of fake leads and receive spam from bots. On top of that, these bots introduce noise to our models in production that rely on user behavioural data. We need to save Adevinta's reputation detecting these fake users.

# Contact

The whole project is pushed into a private GitHub reposotiry (in case the test content is confidential). If you have any questions, difficulties installing or running the project, please reach out to me on <omarsabrout@gmail.com>.

# Remarks

In this section, I explain and show the components of the project alongside the documentation within the code.

## Structure

    .
    ├── config                   # Pre-defined configurations
    │   ├── categories.json      # Must-have categories in input CSV file (Event, Category)
    │   └── params.json          # Parameter grid for GridSearchCV
    ├── log                      # Auto-generated logs
    ├── model                    # Model serialized files (output form training, input for API)
    ├── output                   # Predicted CSV files with columns (UserId, is_fake_probability)
    ├── resources                # Train/Test CSV files
    ├── scripts                  # Source code
    │   ├── api                  # API code
    │   │   ├── api.py           # Run API
    │   │   └── main.py          # API request processinf functions
    │   ├── jupyter              # Notebooks to analyse and visualize datasets
    │   ├── logger.py            # Custom logger
    │   ├── main.py              # Main function to consume a command and predict CSV files
    │   ├── model.py             # Pipeline functions
    │   ├── preprocess.py        # Clean and prepare datasets
    │   └── train.py             # Train a model
    ├── Dockerfile               # Build a docker image of the project
    ├── README.md
    └── requirements.txt         # Python environement packages

Trained models are saved in "/model/" directory with a version number that is represented by a date YYMMDD hence **211212.pkl** and **211212.json**.

- [model_version].pkl is the trained model serialized using the joblib package.

- [model_version].json has the model specifications such as:
  - "params": model assigned parameters (to retrain model if needed).
  - "train_dataset_size": size of the train dataset used in training.
  - "features": feature names for each data example X to be predicted.
  - "precision": precision metric calculated on provided test dataset.
  - "recall": recall metric calculated on provided test dataset.
  - "fb-score": fb-score metric (explained in source code) calculated on provided test dataset.
  - "confusion_matrix": confusion matrix calculated on provided test dataset.

## Feature Engineering

Since our the resulting predictions are regarding each user, I decided to transform the dataset so each row represents a user's log. In order to do so, the dataset tranformed from the format of:

| UserID     |   Event    | Category | Fake |
| ---------- | :--------: | :------: | :--: |
| 21C64F22FC | send_email |   Jobs   |  0   |

To a dataset with a seperate column for each unique pair of (Event, Category) and only one row per user:

| UserID     | send_email_Jobs | click_ad_Leisure | ... | send_sms_Holidays | Fake |
| ---------- | :-------------: | :--------------: | :-: | :---------------: | :--: |
| 21C64F22FC |       23        |        5         | ... |         2         |  0   |

Where each number represents a count of this interaction pair. This way, we adapt the dataset for each user, encode categorical features into digits and prepare our model to give produce a CSV file as required.

## Model Selection

In the vast domain of machine learning, there are many algorithms that can be applied in this case and many of them will do just fine. As a result, I decided to use a classifier from XGBoost since it is usually robust, common to use which is good for debugging and handles binary classification pretty well.

### Oversampling

The provided dataset has a certain imbalance in it regarding positive and negative labels which is pretty common among the topic of bot and fraud detection. A common technique to aid the classifier counter this bias is to perform oversampling on the minority class (positive in this case). Hence, I used SMOTE from the **imblearn** library,

### Parameter Tuning

Since I am no expert on this specific type of datasets and parameter tuning is usually automated in production, I decided to go with a classic **GridSearchCV** over over parameters of the classifier and the oversampler.

## Classification Threashold

As the correlation amon interaction diversity, frequency and fake probability is potentially the most important factor, it was relatively easy to train a model with extremely high f-beta and f-1 scores (usage of f-beta is explaind in source code). Since the test set is seperated from the train set before training the model and NOT used in the training process, it would be faire to assume the simplicity of the targetted function in addiction to the possibility of overfitting. Thus, another test dataset is absolutely needed to verify the issue of overfitting. Currently, the recommended classification threashold would be the default **0.5**.

## Possible Improvements

### API instead of Python package:

A more production friendly approach would be deploying the model in an API that handles the user logs who pass a certain number of interactions (decided by noise analysis from recommender models) within a certain period of time (5 min as in the problem description). This API can then be queried by production servers to blacklist bot users on the fly.

Thus, I implemented an API withint the project to demonstrate this improvement.

### OS Independence

While Python scripts run on any machine that has Python installed, a common practice is adopted in production to use Docker containers since it will probably be deployed on a remote server. These containers require Docker images so

Hence, I implemented a Dockerfile that builds an image (provided Docker is installed on the host machine) using the command:

```bash
docker build -t [image_name] .
```

Where **[image_name]** can be "bot-detection-image" for example. Then we run image using the command:

```bash
docker run --rm -it --entrypoint bash [image_name]
```

After running the image, we proceed to either run the CSV prediction script or run the API as explained in the coming sections. We can skip the installation section since it is already included in Dockerfile.

# Installation

To install the project, Python 3.8 or higher is required and it is recommended to have an empty Python environment. I personally use Anaconda to handle Python environments. To start, download the provided ZIP archive and extract it. Then, install essential packages by the command:

```bash
cd [project_path]
pip install -r requirements.txt
```

Where **[project_path]** is the absolute path of the extracted archive directory. Now the project is ready to be executed.

# Run CSV Prediction

Copy the target CSV file into the project directory and run command:

```bash
python -m scripts.main [csv_path]
```

For example:

```bash
python -m scripts.main resources/fake_users_test.csv
```

Output prediction will be the "/output/" directory.

# Run API

Run command:

```bash
python -m scripts.api.api
```

API will be running on http://0.0.0.0:8090.

## Example Request

```json
// GET http://0.0.00:8090/fake_probability
// Request body in raw JSON
{
  "log": [
    ["03E7EE785DT", "click_carrousel", "Phone"],
    ["F0F3098683T", "click_ad", "Leisure"],
    ["5064A38F0DT", "click_carrousel", "Phone"],
    ["5C8E90A354T", "click_carrousel", "Motor"],
    ["DC1F29D286T", "send_sms", "Motor"],
    ["2DA8AAA602T", "send_email", "Phone"],
    ["281A8EE211T", "click_carrousel", "Phone"],
    ["F866660C47T", "send_email", "Real_State"],
    ["D2E47FF774T", "click_carrousel", "Phone"],
    ["EED30617D3T", "click_carrousel", "Motor"]
  ]
}
```

## Example Response

```json
{
  "predictions": [
    {
      "UserId": "03E7EE785DT",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "281A8EE211T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "2DA8AAA602T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "5064A38F0DT",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "5C8E90A354T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "D2E47FF774T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "DC1F29D286T",
      "is_fake_probability": "0.00001"
    },
    {
      "UserId": "EED30617D3T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "F0F3098683T",
      "is_fake_probability": "0.00002"
    },
    {
      "UserId": "F866660C47T",
      "is_fake_probability": "0.00001"
    }
  ],
  "metadata": {
    "model_version": 211212,
    "api_version": 211213
  }
}
```
