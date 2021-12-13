# Bot Detection

A marketplace is being attacked by bots that produce fake clicks and leads. Themarketplace reputation might be affected if sellers get tons of fake leads and receive spam from bots. On top of that, these bots introduce noise to our models in production that rely on user behavioural data. We need to save Adevinta's reputation detecting these fake users.

# Remarks

## Feature Engineering

## Model Selection

### Oversampling

### Pipeline

## Classification Threashold

## Possible Improvements

# Installation

# Run

Run command:

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
