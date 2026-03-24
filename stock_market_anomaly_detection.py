import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class StockAnomalyDetector:
    def __init__(
        self,
        tickers,
        start_date="2020-01-01",
        end_date="2025-12-31",
        rolling_window=20,
        contamination="auto",
        random_state=42,
        output_dir="outputs"
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.rolling_window = rolling_window
        self.contamination = contamination
        self.random_state = random_state

        # Output folders
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"
        self.tables_dir = self.output_dir / "tables"
        self.output_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        self.results = {}
        self.summary_df = pd.DataFrame()

    def download_stock_data(self, ticker):
        # Download daily stock data
        df = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Ticker"] = ticker
        return df

    def create_features(self, df):
        # Create features used by the anomaly model
        df = df.copy()
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        df["return"] = df[price_col].pct_change()
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["rolling_volatility"] = df["return"].rolling(self.rolling_window).std()
        df["avg_volume"] = df["Volume"].rolling(self.rolling_window).mean()
        df["abnormal_volume"] = df["Volume"] / df["avg_volume"]
        df["price_range"] = (df["High"] - df["Low"]) / df[price_col]
        df["momentum_5d"] = df[price_col].pct_change(5)

        df = df.dropna().reset_index(drop=True)
        return df, price_col

    def fit_model(self, df, feature_cols):
        # Scale features and fit Isolation Forest
        df = df.copy()
        X = df[feature_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state
        )
        model.fit(X_scaled)

        df["anomaly_label"] = model.predict(X_scaled)   # -1 = anomaly, 1 = normal
        df["anomaly_score"] = model.decision_function(X_scaled)
        df["anomaly_strength"] = -df["anomaly_score"]

        return df, model, scaler

    def process_ticker(self, ticker):
        # Full pipeline for one stock
        df = self.download_stock_data(ticker)
        df, price_col = self.create_features(df)

        feature_cols = [
            "return",
            "log_return",
            "rolling_volatility",
            "abnormal_volume",
            "price_range",
            "momentum_5d"
        ]

        df, model, scaler = self.fit_model(df, feature_cols)

        anomaly_count = int((df["anomaly_label"] == -1).sum())
        rows_analyzed = len(df)
        anomaly_rate = anomaly_count / rows_analyzed if rows_analyzed > 0 else 0

        self.results[ticker] = {
            "data": df,
            "model": model,
            "scaler": scaler,
            "price_col": price_col,
            "feature_cols": feature_cols,
            "summary": {
                "Ticker": ticker,
                "Rows_Analyzed": rows_analyzed,
                "Anomalies_Detected": anomaly_count,
                "Anomaly_Rate": round(anomaly_rate, 4)
            }
        }

        return self.results[ticker]

    def run_all(self):
        # Run the pipeline for all stocks
        summaries = []

        for ticker in self.tickers:
            try:
                result = self.process_ticker(ticker)
                summaries.append(result["summary"])
                print(f"Processed {ticker}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        self.summary_df = pd.DataFrame(summaries)

        if not self.summary_df.empty:
            self.summary_df = self.summary_df.sort_values(
                "Anomalies_Detected",
                ascending=False
            ).reset_index(drop=True)

        return self.summary_df

    def get_anomalies(self, ticker, n=None):
        if ticker not in self.results:
            raise ValueError(f"{ticker} has not been processed yet.")

        result = self.results[ticker]
        df = result["data"]
        price_col = result["price_col"]

        anomalies = df.loc[
            df["anomaly_label"] == -1,
            ["Date", "Ticker", price_col, "Volume", "return", "abnormal_volume", "anomaly_score", "anomaly_strength"]
        ].copy()

        anomalies = anomalies.sort_values("anomaly_strength", ascending=False)

        if n is not None:
            anomalies = anomalies.head(n)

        return anomalies

    def print_summary(self):
        if self.summary_df.empty:
            print("No summary available. Run run_all() first.")
            return

        print("\nSummary of anomaly detection across stocks:")
        print(self.summary_df)

    def print_sample_anomalies(self, ticker, n=10):
        print(f"\nSample anomalies for {ticker}:")
        print(self.get_anomalies(ticker, n=n))

    def plot_price_anomalies(self, ticker, save=True, show=True):
        if ticker not in self.results:
            raise ValueError(f"{ticker} has not been processed yet.")

        result = self.results[ticker]
        df = result["data"]
        price_col = result["price_col"]
        anomalies = df[df["anomaly_label"] == -1]

        plt.figure(figsize=(14, 6))
        plt.plot(df["Date"], df[price_col], label=f"{ticker} Price")
        plt.scatter(anomalies["Date"], anomalies[price_col], s=40, label="Anomalies")
        plt.title(f"{ticker} Stock Price with Detected Anomalies")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(self.charts_dir / f"{ticker.lower()}_price_anomalies.png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_anomaly_scores(self, ticker, save=True, show=True):
        if ticker not in self.results:
            raise ValueError(f"{ticker} has not been processed yet.")

        df = self.results[ticker]["data"]

        plt.figure(figsize=(14, 5))
        plt.plot(df["Date"], df["anomaly_score"])
        plt.axhline(0, linestyle="--")
        plt.title(f"{ticker} Anomaly Scores Over Time")
        plt.xlabel("Date")
        plt.ylabel("Decision Function Score")
        plt.tight_layout()

        if save:
            plt.savefig(self.charts_dir / f"{ticker.lower()}_anomaly_scores.png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def save_summary_csv(self):
        if self.summary_df.empty:
            print("Summary is empty. Nothing to save.")
            return

        self.summary_df.to_csv(self.tables_dir / "summary.csv", index=False)

    def save_anomalies_csv(self, ticker):
        anomalies = self.get_anomalies(ticker, n=None)
        anomalies.to_csv(self.tables_dir / f"{ticker.lower()}_anomalies.csv", index=False)

    def save_processed_data_csv(self, ticker):
        if ticker not in self.results:
            raise ValueError(f"{ticker} has not been processed yet.")

        df = self.results[ticker]["data"]
        df.to_csv(self.tables_dir / f"{ticker.lower()}_processed_data.csv", index=False)

    def save_all_outputs(self, show_plots=False):
        if self.summary_df.empty:
            print("No results available. Run run_all() first.")
            return

        self.save_summary_csv()

        for ticker in self.results:
            self.save_anomalies_csv(ticker)
            self.save_processed_data_csv(ticker)
            self.plot_price_anomalies(ticker, save=True, show=show_plots)
            self.plot_anomaly_scores(ticker, save=True, show=show_plots)


if __name__ == "__main__":
    tickers = ["JPM", "GS", "AAPL", "TSLA", "XOM", "AMZN"]

    detector = StockAnomalyDetector(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2025-12-31",
        rolling_window=20,
        contamination="auto",
        output_dir="outputs"
    )

    detector.run_all()
    detector.print_summary()
    detector.print_sample_anomalies("JPM", n=10)
    detector.save_all_outputs(show_plots=False)