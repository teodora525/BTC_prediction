# ₿ Bitcoin Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-4.x-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

*An end-to-end Bitcoin price prediction system powered by ARIMA, LLMs, and automated email alerts.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [License](#-license)

---

## 🔭 Overview

**BTC Prediction** is a Python-based application that scrapes historical Bitcoin price data from Yahoo Finance, trains an ARIMA time-series model to forecast future price changes, and leverages a large language model (Falcon-7B via HuggingFace) to compose personalised investment-advice emails delivered directly to users via Gmail SMTP.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Live Data Scraping** | Fetches up-to-date BTC/USD historical data directly from Yahoo Finance |
| 📈 **ARIMA Forecasting** | Auto-selects the best ARIMA order with `pmdarima` and forecasts price changes |
| 🤖 **LLM-Powered Emails** | Uses Falcon-7B Instruct (via LangChain + HuggingFace Hub) to write context-aware investment emails |
| 📧 **Email Notifications** | Sends formatted alerts to any email address through Gmail SMTP |
| 🖥️ **Interactive CLI** | A user-friendly menu to view current price, expected change, or request an email |
| 📓 **Jupyter Notebook** | Exploratory notebook (`btc_prediction.ipynb`) for step-by-step analysis and visualisations |

---

## 🛠 Tech Stack

### Core Language & Environment
- **Python 3.10+**
- **python-dotenv** – secure environment variable management

### Data Collection
- **Selenium** – headless browser automation for scraping Yahoo Finance
- **BeautifulSoup4** – HTML parsing
- **Requests** – HTTP utilities

### Data Processing & Analysis
- **Pandas** – data wrangling and time-series manipulation
- **NumPy** – numerical computation
- **Matplotlib** – data visualisation

### Machine Learning & Forecasting
- **statsmodels** – ACF/PACF analysis and ARIMA modelling
- **pmdarima** – automatic ARIMA order selection (`auto_arima`)
- **scikit-learn** – evaluation metrics

### AI / LLM
- **LangChain** – LLM orchestration framework
- **HuggingFace Hub** (`tiiuae/falcon-7b-instruct`) – generative AI for email content

### Notifications
- **smtplib** (stdlib) – Gmail SMTP email delivery

---

## 📁 Project Structure

```
BTC_prediction/
├── btc_prediction.py       # Main application script
├── btc_prediction.ipynb    # Jupyter Notebook (EDA & model exploration)
├── btcJAN30.csv            # Sample historical BTC dataset
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10** or higher
- Google Chrome + matching [ChromeDriver](https://chromedriver.chromium.org/downloads) on your `PATH`
- A [HuggingFace account](https://huggingface.co/) with an API token
- A Gmail account with an [App Password](https://support.google.com/accounts/answer/185833) enabled

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/teodora525/BTC_prediction.git
cd BTC_prediction

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (see `.env.example` as reference):

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```

> **Note:** Gmail credentials are currently embedded in the script for demonstration purposes. For production use, move them to `.env` and load with `python-dotenv`.

---

## 💻 Usage

```bash
python btc_prediction.py
```

You will be greeted by an interactive menu:

```
===========================================
         Bitcoin price prediction
===========================================
1. Current Bitcoin price
2. Expected Bitcoin price change
3. I want a price-change notification
0. Exit
```

Alternatively, open the Jupyter Notebook for interactive exploration:

```bash
jupyter notebook btc_prediction.ipynb
```

---

## ⚙️ How It Works

```
┌──────────────────────┐
│  Yahoo Finance       │  ← Selenium scrolls & scrapes historical OHLCV data
└────────┬─────────────┘
         │ HTML
         ▼
┌──────────────────────┐
│  BeautifulSoup4      │  ← Parses the price table into a DataFrame
└────────┬─────────────┘
         │ pandas DataFrame
         ▼
┌──────────────────────┐
│  Feature Engineering │  ← Computes daily percentage change
└────────┬─────────────┘
         │ time series
         ▼
┌──────────────────────┐
│  pmdarima auto_arima │  ← Finds optimal (p,d,q) order
└────────┬─────────────┘
         │ predictions
         ▼
┌──────────────────────┐
│  ARIMA Model         │  ← Rolling-window walk-forward forecasting
└────────┬─────────────┘
         │ forecast + RMSE
         ▼
┌──────────────────────┐
│  LangChain + Falcon  │  ← Generates human-readable investment email
└────────┬─────────────┘
         │ email body
         ▼
┌──────────────────────┐
│  Gmail SMTP          │  ← Delivers notification to the user
└──────────────────────┘
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
