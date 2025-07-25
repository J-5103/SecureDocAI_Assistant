import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

def generate_plot_from_excel(file_path: str, question: str) -> str:
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)

    plot_type = "bar"
    if "line" in question.lower(): plot_type = "line"
    elif "scatter" in question.lower(): plot_type = "scatter"

    plt.figure(figsize=(10, 6))
    x_col, y_col = df.columns[:2]

    if plot_type == "bar":
        df.plot(kind="bar", x=x_col, y=y_col)
    elif plot_type == "line":
        df.plot(kind="line", x=x_col, y=y_col)
    elif plot_type == "scatter":
        sns.scatterplot(data=df, x=x_col, y=y_col)

    os.makedirs("outputs", exist_ok=True)
    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("outputs", plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path
