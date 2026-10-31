<div align="center">

# Support Ticket Classifier API

**Reads a customer support message and returns its category, priority and confidence in one API call.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pytest](https://img.shields.io/badge/tests-12%20passing-3FB950?style=flat-square&logo=pytest&logoColor=white)

</div>

## The problem

Support teams receive hundreds of messages a day. Someone has to read each one, decide what it is about and decide how urgent it is before it reaches the right person. This API does that first pass automatically.

## What it does

```
POST /predict   {"text": "my payment failed twice"}
```
```json
{
  "text": "my payment failed twice",
  "category": "PAYMENT",
  "priority": "high",
  "confidence": 0.751
}
```

* **11 categories:** ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION
* **Priority rule:** PAYMENT, REFUND, CANCEL and ACCOUNT tickets are flagged `high`, everything else is `normal`
* **Batch mode:** send up to 100 tickets in one request

## How it works

| Step | Detail |
|:--|:--|
| Data | [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), about 27,000 labelled messages |
| Split | 80% train, 20% test, stratified by category |
| Features | TF-IDF on words and word pairs |
| Model | Logistic Regression, saved with joblib |
| Serving | FastAPI with Pydantic validation, model loaded once at startup |

## Endpoints

| Method | Path | Purpose |
|:--|:--|:--|
| GET | `/health` | Service check |
| POST | `/predict` | Classify one ticket |
| POST | `/predict/batch` | Classify 1 to 100 tickets |

Input text must be 3 to 2000 characters. Empty, whitespace only, oversized or missing input returns `422`.

Interactive docs are at `http://localhost:8000/docs` once the server is running.

## Run it

```bash
git clone https://github.com/sidhantds/ticket-classifier-api.git
cd ticket-classifier-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Retrain the model (downloads the dataset and prints a full classification report):

```bash
python train.py
```

Run the tests:

```bash
pytest
```

## Tests

12 tests cover the happy path (health check, response shape, PAYMENT and CANCEL routed as high priority, batch size) and validation (short, blank, too long, missing and oversized batch input all rejected).

## Project structure

```
ticket-classifier-api/
├── app/main.py              # API, validation, priority rule
├── model/ticket_model.joblib
├── tests/test_api.py
├── train.py                 # training and evaluation
└── requirements.txt
```

---

<div align="center">

Built by [Sidhant Pujara](https://sidhantpujara.vercel.app/)

</div>
