import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import random

st.set_page_config(page_title="Generador de Sudokus", page_icon="🤓")

def is_valid(board, row, col, num):
    # Verificar fila
    if num in board[row]:
        return False
    
    # Verificar columna
    if num in board[:, col]:
        return False
    
    # Verificar subcuadrícula 3x3
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

def fill_board(board):
    for i in range(81):
        row = i // 9
        col = i % 9
        if board[row][col] == 0:
            numbers = list(range(1, 10))
            random.shuffle(numbers)
            for number in numbers:
                if is_valid(board, row, col, number):
                    board[row][col] = number
                    if not any(0 in row for row in board) or fill_board(board):
                        return True
                    board[row][col] = 0
            return False
    return True

def generate_sudoku():
    board = np.zeros((9, 9), dtype=int)
    fill_board(board)
    return board

def remove_numbers(board, difficulty):
    levels = {"Fácil": 30, "Medio": 40, "Difícil": 50, "Maestro": 65}
    if difficulty not in levels:
        return board
    
    board = board.copy()
    attempts = levels[difficulty]
    while attempts > 0:
        row, col = random.randint(0, 8), random.randint(0, 8)
        while board[row, col] == 0:
            row, col = random.randint(0, 8), random.randint(0, 8)
        board[row, col] = 0
        attempts -= 1
    
    return board

def draw_sudoku(board, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(title, fontsize=16)
    ax.matshow(np.ones_like(board) * -1, cmap="gray_r")

    for (i, j), val in np.ndenumerate(board):
        if val != 0:
            ax.text(j, i, int(val), va='center', ha='center')
    
    # Dibujar las líneas gruesas para separar las subcuadrículas
    for i in range(1, 9):
        linewidth = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color='black', linewidth=linewidth)
        ax.axvline(i - 0.5, color='black', linewidth=linewidth)
    
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def main():
    st.title("Generador de Sudoku")
    
    st.subheader("Hecho por: Arian Vazquez Fernandez")

    # Selección de la dificultad
    difficulty = st.selectbox("Selecciona la dificultad", ["Tablero Completo", "Fácil", "Medio", "Difícil", "Maestro"])
    
    # Estado de la sesión para mantener el tablero completo y el tablero con números eliminados
    if "full_board" not in st.session_state:
        st.session_state.full_board = generate_sudoku()
        st.session_state.puzzle_board = st.session_state.full_board.copy()
    
    # Convertir el tablero en un DataFrame de Pandas
    df = pd.DataFrame(st.session_state.puzzle_board, columns=[f'Columna {i+1}' for i in range(9)], index=[f'Fila {i+1}' for i in range(9)])

    # Mostrar el tablero como imagen
    fig_puzzle = draw_sudoku(st.session_state.puzzle_board, f"Tablero de Sudoku (nivel {difficulty})")
    st.write('⚠️ Recuerda presionar el botón de "Nuevo tablero" al cambiar la dificultad. ⚠️')
    buf_puzzle = BytesIO()
    fig_puzzle.savefig(buf_puzzle, format='png')
    st.image(buf_puzzle, caption=f"Tablero de Sudoku (nivel {difficulty})")

    # Crear tres columnas para los botones
    col1, col2, col3 = st.columns(3)

    # Botón para guardar como imagen el tablero con números eliminados
    with col1:
        st.download_button(
            label="Descargar tablero",
            data=buf_puzzle.getvalue(),
            file_name="Tablero_sudoku.png",
            mime="image/png"
        )

    # Botón para guardar como imagen el tablero completo
    with col2:
        fig_solution = draw_sudoku(st.session_state.full_board, f"Respuesta del Sudoku (nivel {difficulty})")
        buf_solution = BytesIO()
        fig_solution.savefig(buf_solution, format='png')
        st.download_button(
            label="Descargar respuesta",
            data=buf_solution.getvalue(),
            file_name="Respuesta_sudoku.png",
            mime="image/png"
        )

    # Botón para generar un nuevo tablero
    with col3:
        if st.button('Nuevo tablero'):
            st.session_state.full_board = generate_sudoku()
            if difficulty != "Tablero Completo":
                st.session_state.puzzle_board = remove_numbers(st.session_state.full_board, difficulty)
            else:
                st.session_state.puzzle_board = st.session_state.full_board.copy()
            st.rerun()

if __name__ == "__main__":
    main()