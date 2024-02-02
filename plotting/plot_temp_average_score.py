import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data(file_path):
    """Reads CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def calculate_row_and_category_averages(df, row_ranges, columns_to_average):
    """Calculates the average score 'k' for each row and the category averages."""
    category_averages = {}
    row_averages_data = []

    for category, (start, end) in row_ranges.items():
        # Ensure start and end are within DataFrame bounds
        if start >= 0 and end <= len(df):
            category_data = df.iloc[start:end][columns_to_average]
            row_averages = category_data.mean(axis=1)  # Average across columns for each row
            category_averages[category] = row_averages.mean()  # Average of row averages
            for i in range(start, end):
                # Ensure i is within DataFrame bounds
                if i < len(df):
                    k = df.iloc[i][columns_to_average].mean()
                    row_averages_data.append((category, i, k))
                else:
                    print(f"Index {i} is out of bounds. Skipping.")
        else:
            print(f"Range {start}-{end} for category '{category}' is out of DataFrame bounds. Skipping.")

    return pd.DataFrame(category_averages, index=['Category Average']).T, pd.DataFrame(row_averages_data, columns=['Category', 'Row Index', 'k'])

def plot_data(category_averages, row_averages_data, categories, columns_to_average):
    """Plots category averages and individual row averages with horizontal lines."""
    plt.figure(figsize=(7, 3.5))

    gate_type_offsets = {'OR Similarity Score': -0.3, 'AND Similarity Score': -0.1, 'NOT XOR Similarity Score': 0.1, 'NOT AND Similarity Score': 0.3}
    category_colors = {'T(0.2,0.2)': 'orange', 'T(0.2,0.7)': 'lightblue', 'T(0.2,1.3)': 'lightgray', 'T(0.2,2.0)': 'green' }
    category_avg_colors = {'T(0.2,0.2)': 'red', 'T(0.2,0.7)': 'darkblue', 'T(0.2,1.3)': 'darkgray', 'T(0.2,2.0)': 'green'}

    # Plotting individual row data with horizontal lines
    for _, row in row_averages_data.iterrows():
        row_x_values, row_y_values = [], []
        for gate in columns_to_average:
            x_value = categories.index(row['Category']) + gate_type_offsets[gate]
            y_value = row['k']
            row_x_values.append(x_value)
            row_y_values.append(y_value)
            plt.scatter(x_value, y_value, color=category_colors[row['Category']], alpha=0.5)
        plt.plot(row_x_values, row_y_values, linestyle='-', color=category_colors[row['Category']], alpha=0.5)

    # Plotting category averages
    for category in categories:
        avg_x_values, avg_y_values = [], []
        for gate, offset in gate_type_offsets.items():
            avg_x_values.append(categories.index(category) + offset)
            avg_y_values.append(category_averages[category])
        plt.plot(avg_x_values, avg_y_values, linestyle='-', color=category_avg_colors[category], alpha=1.0)

    # Setting plot details
    plt.ylabel('Average Score')
    plt.xlabel('Temperature')
    plt.xticks(ticks=range(len(categories)), labels=categories)
    plt.ylim(bottom=None, top=1)
    plt.grid(True)
    plt.show()

# Constants
file_path = 'logic_scores.csv'
row_ranges = {'T(0.2,0.2)': (189, 200), 'T(0.2,0.7)': (202, 214), 'T(0.2,1.3)': (216, 228), 'T(0.2,2.0)': (230, 241) }
columns_to_average = ['OR Similarity Score', 'AND Similarity Score', 'NOT XOR Similarity Score', 'NOT AND Similarity Score']
categories = ['T(0.2,0.2)', 'T(0.2,0.7)', 'T(0.2,1.3)', 'T(0.2,2.0)']

# Main execution
df = read_data(file_path)
category_averages, row_averages_data = calculate_row_and_category_averages(df, row_ranges, columns_to_average)
plot_data(category_averages.iloc[0], row_averages_data, categories, columns_to_average)

