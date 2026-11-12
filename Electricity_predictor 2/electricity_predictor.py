# Electricity Usage Predictor -----> PS No. 61
# Uses Pandas, NumPy and Matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RATE = 6  # Rs. per unit
DEFAULT_FILE = "usage.csv"


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_data(filename):
    # CSV columns: Month, Units (header row optional)
    try:
        df = pd.read_csv(filename, header=None, dtype=str, skip_blank_lines=True)
    except FileNotFoundError:
        print("Error: File not found ->", filename)
        return None
    except Exception as e:
        print("Error: Could not read file:", e)
        return None

    if df.shape[1] != 2:
        print("Error: CSV must have exactly 2 columns (Month, Units). Found",
              df.shape[1], "column(s).")
        return None

    df.columns = ["Month", "Units"]

    # Skip a header row such as "Month,Units" silently
    header_skipped = False
    if len(df) > 0 and not (is_number(df.iloc[0]["Month"]) and is_number(df.iloc[0]["Units"])):
        df = df.iloc[1:]
        header_skipped = True

    total_rows = len(df)

    # Convert to numbers; bad values become NaN
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

    bad = int(df.isnull().any(axis=1).sum())
    if bad > 0:
        print("Warning:", bad, "row(s) with missing or non-numeric values removed.")
        df = df.dropna()

    df = df.reset_index(drop=True)

    print("\n--- Data Summary ---")
    if header_skipped:
        print("Header row detected and skipped.")
    print("Rows read      :", total_rows)
    print("Valid rows     :", len(df))
    if len(df) > 0:
        print("Total units    :", df["Units"].sum())
        print("Average units  :", round(df["Units"].mean(), 2))
        print("Min / Max units:", df["Units"].min(), "/", df["Units"].max())
    return df


def fit_model(df):
    x = df["Month"].values
    y = df["Units"].values

    slope, intercept = np.polyfit(x, y, 1)

    # R2 score
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1 - ss_res / ss_tot

    return slope, intercept, r2


def plot_graph(df, slope, intercept, month, predicted):
    x = df["Month"].values
    y = df["Units"].values

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual Usage")

    # Line from the first month to the predicted month
    line_x = np.linspace(min(x.min(), month), max(x.max(), month), 100)
    plt.plot(line_x, slope * line_x + intercept, color="red", label="Regression Line")
    plt.scatter([month], [predicted], color="green", marker="*", s=150,
                label="Predicted (Month " + str(month) + ")")

    plt.title("Electricity Usage: Actual vs Predicted")
    plt.xlabel("Month")
    plt.ylabel("Units Consumed")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict(df):
    if len(df) < 2:
        print("Error: At least 2 months of data are required for prediction.")
        return

    try:
        month = int(input("Enter month number to predict: "))
    except ValueError:
        print("Error: Please enter a valid whole month number (for example 5).")
        return

    if month < 1:
        print("Error: Month number must be 1 or greater.")
        return

    slope, intercept, r2 = fit_model(df)
    predicted = slope * month + intercept
    bill = predicted * RATE

    print("\nSlope: %.2f | Intercept: %.2f | R2: %.2f" % (slope, intercept, r2))
    print("Predicted units (Month %d): %.2f" % (month, predicted))
    print("Estimated bill: Rs. %.2f" % bill)
    print("Opening chart 'Actual vs Predicted' (close the window to continue)...")

    plot_graph(df, slope, intercept, month, predicted)


def main():
    filename = input("Enter CSV file name (default %s): " % DEFAULT_FILE).strip()
    if filename == "":
        filename = DEFAULT_FILE

    df = load_data(filename)
    if df is None:
        return

    while True:
        print("\n===== Electricity Usage Predictor =====")
        print("1. Predict")
        print("2. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            predict(df)
        elif choice == "2":
            print("Thank you!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
