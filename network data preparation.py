from itertools import combinations
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np

# Read the Excel file that contains texts and elements for evaluation
for_file_path = "C:/Users/falci/Dropbox/___PROJECTS/10_ArchiMED/ArchiMed_AI_lit/Lit review systematic scoping/data/final/FoR_list_V3.xlsx"
df_FoR = pd.read_excel(for_file_path, sheet_name="clean")

# Generate combinations and store in another DataFrame
results = []
for row in df_FoR.itertuples():
    id_part = row.id
    year = row.merged_year
    # Ensure FoR_list is split into individual elements
    elements = row.FoR_list.split(", ")  # Split the comma-separated string into a list
    for comb in combinations(elements, 2):
        results.append({
            'id': id_part,
            'merged_year': year,
            'From': comb[0],
            'To': comb[1]
        })

for row in df_FoR.itertuples():
    id_part = row.id
    year = row.merged_year
    elements = row.FoR_list.split(", ") if isinstance(row.FoR_list, str) else []

    if len(elements) == 1:  # Handle single element cases
        results.append({
            'id': id_part,
            'merged_year': year,
            'From': elements[0],
            'To': elements[0]  # Self-loop for single-element cases
        })

# Convert the results into a DataFrame
df_results = pd.DataFrame(results)

# Display the resulting DataFrame
print(df_results)
df_results.to_excel("FoR_network_data_V3_2025_05_01.xlsx", index=False)

# Count unique IDs in both dataframes
unique_ids_FoR_list = df_FoR['id'].nunique()
unique_ids_FoR_network = df_results['id'].nunique()

# Print the unique ID counts
print(f"Unique IDs in FoR_list_V2: {unique_ids_FoR_list}")
print(f"Unique IDs in FoR_network_data_V2: {unique_ids_FoR_network}")

# Get all unique IDs from both datasets
unique_ids_FoR_list = set(df_FoR['id'].unique())
unique_ids_FoR_network = set(df_results['id'].unique())

# Find missing IDs
missing_ids = unique_ids_FoR_list - unique_ids_FoR_network

print(f"Missing IDs count: {len(missing_ids)}")
print(f"Sample Missing IDs: {list(missing_ids)[:17]}")
#############################################
# PLOT NETWORKS
# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

# Output directory for the PNG files
output_dir = "C:/Users/falci/Dropbox/___PROJECTS/10_ArchiMED/ArchiMed_AI_lit/Lit review systematic scoping/draft/network figures 3"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Calculate node frequencies (centrality) and edge frequencies
node_frequencies = (
    pd.concat([
        df_results[['From']].rename(columns={'From': 'Node'}),
        df_results[['To']].rename(columns={'To': 'Node'})
    ], ignore_index=True)
    .value_counts()
    .reset_index(name='Frequency')
    .set_index('Node')['Frequency']
    .to_dict()
)

edge_frequencies = (
    df_results.groupby(['From', 'To'])
    .size()
    .reset_index(name='Weight')
    .set_index(['From', 'To'])['Weight']
    .to_dict()
)

# Assign a unique color to each node
unique_nodes = list(node_frequencies.keys())
color_map = cm.tab10(range(len(unique_nodes)))  # Generate unique colors
#node_colors = {node: color_map[i] for i, node in enumerate(unique_nodes)}

import numpy as np

node_colors = {
    'Health': np.array([0.0, 0.45, 0.70]),       # Strong Blue
    'Biomed_Clinical': np.array([1.0, 0.498, 0.055]),  # Orange
    'Bio': np.array([0.173, 0.627, 0.173]),       # Green
    'Human': np.array([0.8, 0.47, 0.65]),         # Violet-Pink
    'Chemical': np.array([0.839, 0.153, 0.157]),  # Red
    'Psycho': np.array([0.35, 0.7, 0.9]),         # Sky Blue
    'Hiso_Archaeo': np.array([0.9, 0.6, 0.6]),    # Light Coral
    'Math': np.array([0.6, 0.6, 0.6]),            # Neutral Gray
    'Physic': np.array([0.0, 0.6, 0.2]),          # Forest Green
    'Arts': np.array([0.58, 0.4, 0.74]),          # Purple
    'Agr_Vet_Food': np.array([1.0, 0.75, 0.0]),   # Bright Yellow-Orange
    'Lang': np.array([0.36, 0.6, 0.8]),           # Light Blue
    'Comput': np.array([1.0, 0.6, 0.8]),          # Pink
    'Eng': np.array([0.3, 0.8, 0.6]),             # Aquamarine
    'Eco': np.array([0.9, 0.3, 0.2]),             # Rust
    'Educ': np.array([0.65, 0.35, 0.9]),          # Lavender Purple
    'Indig': np.array([0.3, 0.9, 0.3]),           # Bright Green
    'Philo': np.array([0.5, 0.7, 0.1]),           # Avocado
    'Envir': np.array([0.1, 0.7, 0.5]),           # Teal
    'Law': np.array([0.95, 0.2, 0.6]),            # Hot Pink
    'Com': np.array([0.7, 0.7, 0.1])              # Mustard
}


# Get unique years
unique_years = df_results['merged_year'].unique()

# Create a graph for each year and save as PNG
for year in unique_years:
    # Filter DataFrame by year
    df_year = df_results[df_results['merged_year'] == year]

    # Create a graph for the current year
    G = nx.Graph()
    for _, row in df_year.iterrows():
        G.add_edge(row['From'], row['To'], weight=edge_frequencies[(row['From'], row['To'])])

    # Check if Biomed_Clinical exists in the graph
    fixed_nodes = ['Biomed_Clinical'] if 'Biomed_Clinical' in G.nodes else []
    fixed_pos = {node: [0, 0] for node in fixed_nodes}

    # Calculate positions
    if fixed_nodes:  # If there are fixed nodes
        pos = nx.spring_layout(G, seed=42, k=0.15, pos=fixed_pos, fixed=fixed_nodes)
    else:  # No fixed nodes
        pos = nx.spring_layout(G, seed=42, k=0.15)  # Adjust k to control node spacing

    # Set a fixed figure size
    plt.figure(figsize=(14, 12))  # Increased width to make space for the legend

    # Draw nodes with sizes proportional to their frequency and unique colors
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_frequencies[node] * 0.5 for node in G.nodes],  # Scale node sizes
        node_color=[node_colors[node] for node in G.nodes]
    )

    # Draw edges with widths proportional to their weight
    nx.draw_networkx_edges(
        G, pos,
        width=[G[u][v]['weight'] * 0.005 for u, v in G.edges],  # Scale edge widths
        edge_color='gray'
    )

    # Draw node labels (disabled to reduce clutter on the graph)
    # nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    # Title for the graph
    plt.title(f"Year {year}", fontsize=16)

    # Add a legend with node names
    sorted_nodes = sorted(G.nodes)  # Sort nodes alphabetically
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=node,
               markersize=10, markerfacecolor=node_colors[node])
        for node in sorted_nodes
    ]
    plt.legend(handles=legend_handles, title="Node Legend", loc="center left", bbox_to_anchor=(1, 0.5))

    # Save the plot as a PNG file
    file_path = os.path.join(output_dir, f"network_{year}.png")
    plt.savefig(file_path, format='png', dpi=300, bbox_inches="tight")  # Adjust to include the legend
    plt.close()  # Close the figure to avoid displaying it

    # Save as EPS
    eps_path = os.path.join(output_dir, f"network_{year}.eps")
    plt.savefig(eps_path, format='eps', dpi=300, bbox_inches="tight")

    print(f"Saved network graph for year {year} to {file_path}")

# Descriptive stats of network
import pandas as pd
import networkx as nx

# Define the range of years
start_year = 1916
end_year = 2024

# Filter the DataFrame to include only the relevant years
df_filtered = df_results[df_results['merged_year'].between(start_year, end_year)]

# Output file
output_file = "network_statistics_by_year_2025_05_01.xlsx"

# Initialize an Excel writer
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Loop through each year
    for specific_year in range(start_year, end_year + 1):
        # Filter the DataFrame by the specific year
        df_year = df_filtered[df_filtered['merged_year'] == specific_year]

        # Skip if there is no data for the year
        if df_year.empty:
            print(f"No data for the year {specific_year}. Skipping...")
            continue

        # Create a graph for the specific year
        G = nx.Graph()
        for _, row in df_year.iterrows():
            # Add edges with weights from edge_frequencies, defaulting to 1 if not found
            G.add_edge(
                row['From'], row['To'],
                weight=edge_frequencies.get((row['From'], row['To']), 1)
            )

        # Calculate network statistics
        degree_dict = dict(G.degree())
        avg_clustering = nx.average_clustering(G)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        unique_id_count = df_year['id'].nunique()

        # Create the summary row as a DataFrame
        summary_df = pd.DataFrame([{
            'Summary': 'Summary',
            'Nodes': f"Nodes: {num_nodes}",
            'Edges': f"Edges: {num_edges}",
            'Avg Clustering': f"Avg Clustering: {avg_clustering:.4f}",
            'Unique IDs': f"Unique IDs: {unique_id_count}"
        }])

        # Write the summary to a sheet named for the year
        sheet_name = f"Year_{specific_year}"
        summary_df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"Summary for year {specific_year} written to the sheet '{sheet_name}'.")


# Descriptive stats of network
import pandas as pd
import networkx as nx

# Define the range of years
start_year = 1916
end_year = 2024

# Filter the DataFrame to include only the relevant years
df_filtered = df_results[df_results['merged_year'].between(start_year, end_year)]

# Output file
output_file = "network_statistics_by_year_2025_05_01.xlsx"

# Initialize an Excel writer
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Loop through each year
    for specific_year in range(start_year, end_year + 1):
        # Filter the DataFrame by the specific year
        df_year = df_filtered[df_filtered['merged_year'] == specific_year]

        # Skip if there is no data for the year
        if df_year.empty:
            print(f"No data for the year {specific_year}. Skipping...")
            continue

        # Create a graph for the specific year
        G = nx.Graph()
        for _, row in df_year.iterrows():
            # Add edges with weights from edge_frequencies, defaulting to 1 if not found
            G.add_edge(
                row['From'], row['To'],
                weight=edge_frequencies.get((row['From'], row['To']), 1)
            )

        # Calculate network statistics
        degree_dict = dict(G.degree())
        degree_centrality = nx.degree_centrality(G)
        katz_centrality = nx.katz_centrality_numpy(G, alpha=0.01, beta=1.0)
        clustering_coefficient = nx.clustering(G)
        betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
        avg_clustering = nx.average_clustering(G)

        # Overall metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        unique_id_count = df_year['id'].nunique()

        # Save the statistics to a DataFrame
        stats_df = pd.DataFrame({
            'Node': degree_dict.keys(),
            'Degree': degree_dict.values(),
            'Degree Centrality': degree_centrality.values(),
            'Katz Centrality': katz_centrality.values(),
            'Clustering Coefficient': clustering_coefficient.values(),
            'Betweenness Centrality': betweenness_centrality.values()
        })

        # Create the summary row as a DataFrame
        summary_df = pd.DataFrame([{
            'Node': 'Summary',
            'Degree': f"Nodes: {num_nodes}",
            'Degree Centrality': f"Edges: {num_edges}",
            'Katz Centrality': None,
            'Clustering Coefficient': f"Avg Clustering: {avg_clustering:.4f}",
            'Betweenness Centrality': f"Unique IDs: {unique_id_count}"
        }])

        # Concatenate the statistics and summary rows
        stats_df = pd.concat([stats_df, summary_df], ignore_index=True)

        # Write to a sheet named for the year
        sheet_name = f"Year_{specific_year}"
        stats_df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"Statistics for year {specific_year} written to the sheet '{sheet_name}'.")

#############################################################
# Descriptive stats of network in longitudinal format
# Descriptive stats of network in longitudinal format
import pandas as pd
import networkx as nx
from collections import Counter

# Define the range of years
start_year = 1916
end_year = 2024

# Filter the DataFrame to include only the relevant years
df_filtered = df_results[df_results['merged_year'].between(start_year, end_year)]

# Compute edge frequencies
edges = df_filtered.apply(lambda row: tuple(sorted([row['From'], row['To']])), axis=1)
edge_frequencies = Counter(edges)  # Count frequencies of edges

# Output file for the longitudinal format
longitudinal_output_file = "network_longitudinal_statistics_with_network_metrics_and_edge_frequency_2025_05_02.xlsx"

# List to store longitudinal data
longitudinal_data = []

# Loop through each year
for specific_year in range(start_year, end_year + 1):
    # Filter the DataFrame by the specific year
    df_year = df_filtered[df_filtered['merged_year'] == specific_year]

    # Skip if there is no data for the year
    if df_year.empty:
        print(f"No data for the year {specific_year}. Skipping...")
        continue

    # Create a graph for the specific year
    G = nx.Graph()
    for _, row in df_year.iterrows():
        # Add edges with weights based on edge frequencies
        edge = tuple(sorted([row['From'], row['To']]))
        G.add_edge(edge[0], edge[1], weight=edge_frequencies.get(edge, 1))

    # Calculate network statistics
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    katz_centrality = nx.katz_centrality_numpy(G, alpha=0.01, beta=1.0)
    clustering_coefficient = nx.clustering(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    avg_clustering = nx.average_clustering(G)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree_centrality = sum(degree_centrality.values()) / num_nodes if num_nodes > 0 else 0
    avg_katz_centrality = sum(katz_centrality.values()) / num_nodes if num_nodes > 0 else 0
    avg_betweenness_centrality = sum(betweenness_centrality.values()) / num_nodes if num_nodes > 0 else 0
    network_density = nx.density(G)

    # Compute diameter and avg path length using the largest connected component
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_lcc = G.subgraph(largest_cc_nodes).copy()
            diameter = nx.diameter(G_lcc)
            avg_path_length = nx.average_shortest_path_length(G_lcc)
    except nx.NetworkXError:
        diameter = float('inf')
        avg_path_length = float('inf')

    # Count unique IDs for the year
    unique_id_count = df_year['id'].nunique()

    # Populate longitudinal data for each node
    for node in G.nodes():
        longitudinal_data.append({
            'Year': specific_year,
            'Field': node,
            'Degree': degree_dict.get(node, 0),
            'Degree Centrality': degree_centrality.get(node, 0),
            'Katz Centrality': katz_centrality.get(node, 0),
            'Clustering Coefficient': clustering_coefficient.get(node, 0),
            'Betweenness Centrality': betweenness_centrality.get(node, 0),
            'Number of Nodes': num_nodes,
            'Number of Edges': num_edges,
            'Network Diameter': diameter,
            'Network Density': network_density,
            'Average Degree Centrality': avg_degree_centrality,
            'Average Katz Centrality': avg_katz_centrality,
            'Average Clustering Coefficient': avg_clustering,
            'Average Betweenness Centrality': avg_betweenness_centrality,
            'Average Path Length': avg_path_length,
            'Number of Unique IDs': unique_id_count
        })

# Create a DataFrame from the longitudinal data
longitudinal_df = pd.DataFrame(longitudinal_data)

# Export the DataFrame to Excel
longitudinal_df.to_excel(longitudinal_output_file, index=False)
print(f"Longitudinal data exported to {longitudinal_output_file}.")

#######################
import matplotlib.pyplot as plt
import seaborn as sns

# Filter for a few example fields to keep the plot clear
selected_fields = longitudinal_df['Field'].unique()[:5]  # Change this to select specific fields

for metric in ['Degree', 'Degree Centrality', 'Katz Centrality', 'Clustering Coefficient', 'Betweenness Centrality']:
    plt.figure(figsize=(12, 6))
    for field in selected_fields:
        field_data = longitudinal_df[longitudinal_df['Field'] == field]
        plt.plot(field_data['Year'], field_data[metric], label=f"{field}")
    plt.title(f"Evolution of {metric} Over Time for Selected Fields")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.legend(title="Fields")
    plt.grid(True)
    output_file = f"{metric}_evolution_selected_fields.png"
    plt.savefig(output_file)  # Save plot as an image
    plt.close()
    print(f"Saved node-level metric plot: {output_file}")



# Extract unique network-level statistics per year
network_metrics = longitudinal_df[['Year', 'Number of Nodes', 'Number of Edges', 'Network Diameter',
                                    'Network Density', 'Average Degree Centrality', 'Average Katz Centrality',
                                    'Average Clustering Coefficient', 'Average Betweenness Centrality']].drop_duplicates()

for metric in ['Number of Nodes', 'Number of Edges', 'Network Diameter', 'Network Density',
               'Average Degree Centrality', 'Average Katz Centrality', 'Average Clustering Coefficient',
               'Average Betweenness Centrality']:
    plt.figure(figsize=(12, 6))
    plt.plot(network_metrics['Year'], network_metrics[metric], marker='o', label=metric)
    plt.title(f"Evolution of {metric} Over Time")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.grid(True)
    output_file = f"{metric}_evolution_network_metrics.png"
    plt.savefig(output_file)  # Save plot as an image
    plt.close()
    print(f"Saved network-level metric plot: {output_file}")

#// DYNAMIC GRAPH
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib.colors import to_hex
import numpy as np

# Prepare data
unique_years = sorted(df_results['merged_year'].unique())

# Determine the largest network's bounding box
largest_x_range = [-1, 1]
largest_y_range = [-1, 1]

for year in unique_years:
    # Filter the data for the current year
    df_year = df_results[df_results['merged_year'] == year]

    # Create a graph for the year
    G = nx.Graph()
    for _, row in df_year.iterrows():
        G.add_edge(row['From'], row['To'], weight=edge_frequencies.get((row['From'], row['To']), 1))

    # Force Biomed_Clinical to be at the center if it exists
    fixed_nodes = ['Biomed_Clinical'] if 'Biomed_Clinical' in G.nodes else []
    fixed_pos = {node: [0, 0] for node in fixed_nodes}

    # Calculate positions
    pos = nx.spring_layout(G, seed=42, k=1.2, pos=fixed_pos, fixed=fixed_nodes)

    # Update largest ranges
    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    largest_x_range = [min(largest_x_range[0], min(x_coords)), max(largest_x_range[1], max(x_coords))]
    largest_y_range = [min(largest_y_range[0], min(y_coords)), max(largest_y_range[1], max(y_coords))]

# Normalize the largest range and zoom in by reducing the range slightly
zoom_factor = 0.8  # Adjust zoom level (smaller = more zoomed in)
x_center = (largest_x_range[1] + largest_x_range[0]) / 2
y_center = (largest_y_range[1] + largest_y_range[0]) / 2
largest_x_range = [(x - x_center) * zoom_factor + x_center for x in largest_x_range]
largest_y_range = [(y - y_center) * zoom_factor + y_center for y in largest_y_range]

x_scale = 2 / (largest_x_range[1] - largest_x_range[0])
y_scale = 2 / (largest_y_range[1] - largest_y_range[0])

# Generate frames for animation
frames = []

for year in unique_years:
    # Filter the data for the current year
    df_year = df_results[df_results['merged_year'] == year]

    # Create a graph for the year
    G = nx.Graph()
    for _, row in df_year.iterrows():
        G.add_edge(row['From'], row['To'], weight=edge_frequencies.get((row['From'], row['To']), 1))

    # Force Biomed_Clinical to be at the center if it exists
    fixed_nodes = ['Biomed_Clinical'] if 'Biomed_Clinical' in G.nodes else []
    fixed_pos = {node: [0, 0] for node in fixed_nodes}

    # Calculate positions
    pos = nx.spring_layout(G, seed=42, k=1.2, pos=fixed_pos, fixed=fixed_nodes)

    # Normalize positions using the largest ranges
    pos = {node: [
        (p[0] - largest_x_range[0]) * x_scale - 1,
        (p[1] - largest_y_range[0]) * y_scale - 1
    ] for node, p in pos.items()}

    # Prepare node and edge data for Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_labels = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        frequency = max(1, node_frequencies.get(node, 1))
        node_size.append(np.log(frequency + 1) * 8)  # Logarithmic scaling for node size
        node_color.append(to_hex(node_colors[node]))  # Convert matplotlib color to hex
        node_labels.append(node)

    # Create a frame for the year
    frames.append(go.Frame(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=0.8), hoverinfo='none'),
            go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                text=node_labels,
                textposition="top center",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    showscale=False,
                    line=dict(color='black', width=0.5)
                ),
                hoverinfo='text'
            )
        ],
        name=str(year)
    ))

# Create the initial figure layout
fig = go.Figure(
    data=[
        go.Scatter(x=[], y=[], mode='lines', line=dict(color='gray', width=1), hoverinfo='none'),
        go.Scatter(
            x=[], y=[], mode='markers+text',
            text=[],
            textposition="top center",
            marker=dict(
                size=[],
                color=[],
                showscale=False,
                line=dict(color='black', width=0.5)
            ),
            hoverinfo='text'
        )
    ],
    layout=go.Layout(
        title="Evolution of the knowledge network of neurosyphilis research",
        xaxis=dict(showgrid=False, zeroline=False, range=[-1.75, 0.5]),  # Fixed range for the largest network
        yaxis=dict(showgrid=False, zeroline=False, range=[-1.5, 1.25]),  # Fixed range for the largest network
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]),  # Slower transition
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )
        ]
    ),
    frames=frames
)

# Add a slider to control the year
fig.update_layout(
    sliders=[dict(
        steps=[
            dict(method="animate", args=[[str(year)], dict(frame=dict(duration=1000, redraw=True), mode="immediate")], label=str(year))  # Slower transition
            for year in unique_years
        ],
        transition=dict(duration=500),
        x=0.1,
        xanchor="left",
        y=0,
        currentvalue=dict(font=dict(size=16), prefix="Year: ", visible=True, xanchor="right"),
        len=0.9
    )]
)

# Export to HTML
output_html_file = "dynamic_network_biomed_centered_zoomed.html"
fig.write_html(output_html_file)
print(f"Dynamic network visualization exported to {output_html_file}.")
