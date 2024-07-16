import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from queue import PriorityQueue
from io import BytesIO

st.set_page_config(page_title="Maze Generator", page_icon="🌀")

def create_maze(width, height):
    maze = np.ones((height * 2 + 1, width * 2 + 1))
    
    def get_unvisited_neighbors(x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny * 2 + 1, nx * 2 + 1] == 1:
                neighbors.append((nx, ny))
        return neighbors
    
    stack = [(0, 0)]
    maze[1, 1] = 0
    while stack:
        x, y = stack[-1]
        neighbors = get_unvisited_neighbors(x, y)
        if neighbors:
            nx, ny = neighbors[np.random.randint(len(neighbors))]
            maze[y + ny + 1, x + nx + 1] = 0
            maze[ny * 2 + 1, nx * 2 + 1] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    
    end = (width * 2 - 2, height * 2 - 2)
    maze[end] = 0
    return maze

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end):
    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and maze[ny, nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, end)}
    explored = set()

    while not pq.empty():
        current = pq.get()[1]
        explored.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, end)
                pq.put((f_score[neighbor], neighbor))

    return None, explored

def plot_maze(maze, path=None, explored=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap='binary')
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='black'))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='white'))
    
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=True, color='green'))
    ax.add_patch(Rectangle((maze.shape[1] - 2.5, maze.shape[0] - 2.5), 1, 1, fill=True, color='limegreen'))
    
    if explored:
        for cell in explored:
            if cell != (1, 1) and cell != (maze.shape[1] - 2, maze.shape[0] - 2):
                ax.add_patch(Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, fill=True, color='lightskyblue', alpha=0.5))
    
    if path:
        for cell in path:
            if cell != (1, 1) and cell != (maze.shape[1] - 2, maze.shape[0] - 2):
                ax.add_patch(Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, fill=True, color='red', alpha=0.5))
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def main():
    st.title("Maze Generator and Solver (using A*)")
    
    st.subheader("Made by: Arian Vazquez Fernandez")

    maze_size = st.slider("Size of the maze", min_value=5, max_value=60, value=15, step=1)
    
    if "maze" not in st.session_state:
        st.session_state.maze = create_maze(maze_size, maze_size)
        st.session_state.path = None
        st.session_state.explored = None

    show_solution = st.checkbox("Show solution")

    st.write('⚠️ Remember to press the "New" button when changing the size or show solution of the maze. ⚠️')

    st.write('⚠️ Big mazes can take more time to generate and solve. ⚠️')

    fig = plot_maze(st.session_state.maze, st.session_state.path if show_solution else None, st.session_state.explored if show_solution else None)
    st.pyplot(fig)

    st.write(f"Maze size {maze_size}, renember to conect the green points.")

    col1, col2, col3 = st.columns(3)

    with col1:
        buf = BytesIO()
        fig.savefig(buf, format='png')
        st.download_button(
            label="Download",
            data=buf.getvalue(),
            file_name="labyrinth.png",
            mime="image/png"
        )

    with col2:
        if st.button('Solve'):
            start = (1, 1)
            end = (st.session_state.maze.shape[1] - 2, st.session_state.maze.shape[0] - 2)
            st.session_state.path, st.session_state.explored = astar(st.session_state.maze, start, end)
            st.rerun()

    with col3:
        if st.button('New'):
            st.session_state.maze = create_maze(maze_size, maze_size)
            st.session_state.path = None
            st.session_state.explored = None
            st.rerun()

if __name__ == "__main__":
    main()