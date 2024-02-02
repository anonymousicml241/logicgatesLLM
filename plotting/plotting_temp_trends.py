import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def read_data(file_path):
    """Reads CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def calculate_averages(df, row_ranges, columns_to_average):
    """Calculates averages for specified row ranges and columns."""
    averages = {}
    for category, (start, end) in row_ranges.items():
        averages[category] = df.iloc[start:end][columns_to_average].mean()
    return pd.DataFrame(averages)

def prepare_scatter_data(averages_df):
    """Prepares data for clustering scatter points."""
    scatter_data = averages_df.T.reset_index().rename(columns={'index': 'Temperature'})
    melted_scatter_data = scatter_data.melt(id_vars='Temperature', var_name='Gate Type', value_name='Average Score')
    #Generic
    #melted_scatter_data['Temperature'] = pd.Categorical(melted_scatter_data['Temperature'], ["Opinion", "Factual", "Imaginative"])
    #Hallucinations
    melted_scatter_data['Temperature'] = pd.Categorical(melted_scatter_data['Temperature'], ["T(0.2,0.2)", "T(0.2,0.7)", "T(0.2,1.3)", "T(0.2,2.0)"])
    return melted_scatter_data

def plot_data(df, scatter_data_expanded, categories_ranges, gate_type_offsets, gate_type_styles, gate_type_colors):
    """Plots the average and individual logic scores."""
    plt.figure(figsize=(7, 3.5) )

    # Assigning x_values for scatter plot
    scatter_data_expanded['x_value'] = scatter_data_expanded['Temperature'].cat.codes
    for gate_type, offset in gate_type_offsets.items():
        scatter_data_expanded.loc[scatter_data_expanded['Gate Type'] == gate_type, 'x_value'] += offset

    # Plotting average scores
    for prompt_type in scatter_data_expanded['Temperature'].cat.categories:
        subset = scatter_data_expanded[scatter_data_expanded['Temperature'] == prompt_type]
        sns.scatterplot(data=subset, x='x_value', y='Average Score', hue='Gate Type', style='Gate Type', s=100, legend=False)
        plt.plot(subset['x_value'], subset['Average Score'], linestyle='-', alpha=1)

    # Plotting individual data points with faint lines
    for category, (start, end) in categories_ranges.items():
        category_data = df.iloc[start:end]
        for idx, row in category_data.iterrows():
            row_x_values, row_y_values = [], []
            for gate_type, offset_value in gate_type_offsets.items():
                x_value = scatter_data_expanded['Temperature'].cat.categories.get_loc(category) + offset_value
                y_value = row[gate_type]
                row_x_values.append(x_value)
                row_y_values.append(y_value)
            plt.plot(row_x_values, row_y_values, linestyle='-', color='gray', alpha=0.2)

    # Custom legend entries
    legend_elements = [plt.Line2D([0], [0], marker=style, color='w', label=gate, markersize=10, markerfacecolor=color)
                       for gate, style, color in zip(gate_type_styles.keys(), gate_type_styles.values(), gate_type_colors.values())]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.63, 1.1), fontsize='small')


    # Setting plot details
    #plt.title('Average and Individual Logic Scores')
    plt.ylim(0, 1)
    plt.ylabel('Lole Network Score')
    plt.xlabel('Temperature Variations')
    # Generic
    #plt.xticks(ticks=[0, 1, 2], labels=["Opinion", "Factual", "Imaginative"])
    # Hallucinations
    plt.xticks(ticks=[0, 1, 2, 3], labels=["T=0.2", "T=0.7", "T=1.3", "T=2.0"])
    plt.grid(True)
    plt.show()

# Constants
file_path = 'logic_scores.csv'
#Response Types
#row_ranges = {'Opinion': (1, 12), 'Factual': (59, 70), 'Imaginative': (83, 94)}
#Hallucinations
row_ranges = {'T(0.2,0.2)': (188, 196), 'T(0.2,0.7)': (198, 206), 'T(0.2,1.3)': (208, 216), 'T(0.2,2.0)': (218, 226) }
columns_to_average = ['OR Similarity Score', 'AND Similarity Score', 'NOT XOR Similarity Score', 'NOT AND Similarity Score']
gate_type_offsets = {'OR Similarity Score': -1.5 * 0.1, 'AND Similarity Score': -0.5 * 0.1, 
                     'NOT XOR Similarity Score': 0.5 * 0.1, 'NOT AND Similarity Score': 1.5 * 0.1}
gate_type_styles = {'OR Similarity Score': 'o', 'AND Similarity Score': 'X', 
                    'NOT XOR Similarity Score': '^', 'NOT AND Similarity Score': 's'}
gate_type_colors = {'OR Similarity Score': 'blue', 'AND Similarity Score': 'orange', 
                    'NOT XOR Similarity Score': 'green', 'NOT AND Similarity Score': 'red'}

# Main execution
df = read_data(file_path)
averages_df = calculate_averages(df, row_ranges, columns_to_average)
scatter_data_expanded = prepare_scatter_data(averages_df)
plot_data(df, scatter_data_expanded, row_ranges, gate_type_offsets, gate_type_styles, gate_type_colors)
