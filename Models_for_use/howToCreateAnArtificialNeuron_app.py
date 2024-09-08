import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron

# Set page config
st.set_page_config(page_title="Logic Gate Neurons", page_icon="🧠", layout="wide")

# Title and introduction
st.title("Logic Gate Neurons: Learning in Action 🧠")
st.write("""
Explore how artificial neurons learn to mimic logic gates! This app demonstrates the learning process
of neurons as they attempt to replicate AND, OR, and XOR gates. Watch as the neuron's understanding
evolves with each generation.
""")

# Function to create decision boundary plot with colored points
def plot_decision_boundary(classifier, gate_type, data, labels):
    x_values = np.linspace(-0.5, 1.5, 100)
    y_values = np.linspace(-0.5, 1.5, 100)
    
    # Create point grid without using itertools
    point_grid = []
    for x in x_values:
        for y in y_values:
            point_grid.append([x, y])
    
    distances = classifier.decision_function(point_grid)
    distances_matrix = np.reshape(distances, (100, 100))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.pcolormesh(x_values, y_values, distances_matrix, cmap='coolwarm', alpha=0.8)
    plt.colorbar(heatmap)
    
    # Plot the decision boundary
    ax.contour(x_values, y_values, distances_matrix, levels=[0], colors='k', linestyles='--')
    
    # Plot the data points
    for (x, y), label in zip(data, labels):
        color = 'green' if label == 1 else 'red'
        ax.scatter(x, y, c=color, s=200, edgecolor='black', linewidth=1.5, zorder=5)
        ax.annotate(f'({x},{y})', (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax.set_title(f'Decision Boundary for {gate_type} Gate')
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    return fig

# Logic gate data
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = {
    'AND': [0, 0, 0, 1],
    'OR': [0, 1, 1, 1],
    'XOR': [0, 1, 1, 0]
}

# Sidebar for gate selection
gate_type = st.sidebar.selectbox("Select Logic Gate", ["AND", "OR", "XOR"])

# Initialize session state
if 'generation' not in st.session_state:
    st.session_state.generation = 1

# Function to update generation
def update_generation():
    st.session_state.generation += 1

# Create and train the perceptron
perceptron = Perceptron(max_iter=st.session_state.generation, random_state=42)
perceptron.fit(data, labels[gate_type])

# Plot decision boundary with colored points
st.pyplot(plot_decision_boundary(perceptron, gate_type, data, labels[gate_type]))

# Display current generation
st.subheader(f"Current Generation: {st.session_state.generation}")

# Calculate and display accuracy
accuracy = perceptron.score(data, labels[gate_type]) * 100
st.metric("Accuracy", f"{accuracy:.2f}%")

if gate_type == "XOR" and st.session_state.generation > 5:
    st.warning("""
    Note: A single perceptron cannot learn the XOR function perfectly. 
    This is because XOR is not linearly separable. To solve XOR, we need 
    a multi-layer neural network with at least one hidden layer.
    """)
elif accuracy == 100:
    st.success("The neuron has successfully learned the logic gate function!")
else:
    st.info("Keep advancing generations to see if the neuron can improve its accuracy.")

# Button to advance to next generation
if st.button("Advance to Next Generation"):
    update_generation()
    st.experimental_rerun()

# Reset button
if st.button("Reset to Generation 1"):
    st.session_state.generation = 1
    st.experimental_rerun()

# Explanations
st.subheader("Understanding the Visualization")
st.write("""
The graph above shows how the neuron is trying to learn the logic gate function:
- Green points represent outputs of 1 (true)
- Red points represent outputs of 0 (false)
- The background color represents the neuron's decision: blue for 0, red for 1
- The dashed line is the decision boundary: it's where the neuron switches from outputting 0 to 1

For a neuron to successfully learn a logic gate, it needs to find a line (decision boundary) that correctly separates the red and green points.
""")

if gate_type == "AND":
    st.write("""
    For the AND gate:
    - Only the point (1,1) should be on the "1" (red) side of the line
    - All other points should be on the "0" (blue) side
    - This is possible with a single line, so a single neuron can learn it perfectly
    """)
elif gate_type == "OR":
    st.write("""
    For the OR gate:
    - The points (0,1), (1,0), and (1,1) should be on the "1" (red) side of the line
    - Only the point (0,0) should be on the "0" (blue) side
    - This is possible with a single line, so a single neuron can learn it perfectly
    """)
elif gate_type == "XOR":
    st.write("""
    For the XOR gate:
    - The points (0,1) and (1,0) should be on the "1" (red) side of the line
    - The points (0,0) and (1,1) should be on the "0" (blue) side
    - This is impossible to achieve with a single straight line!
    - That's why a single neuron can't learn XOR perfectly - it needs multiple neurons (a multi-layer network) to solve this problem
    """)

# Display truth table
st.subheader("Truth Table")
truth_table = pd.DataFrame({
    'Input 1': [0, 0, 1, 1],
    'Input 2': [0, 1, 0, 1],
    'Output': labels[gate_type]
})
st.table(truth_table)