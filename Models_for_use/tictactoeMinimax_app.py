import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮")

# Inicialización de sesión
if 'board' not in st.session_state:
    st.session_state.board = [["" for _ in range(3)] for _ in range(3)]  # Tablero vacío
    st.session_state.winner = None
    st.session_state.current_player = 'X'

# Parámetros
levelOfFkup = st.number_input('Depth of the decision tree for AI predictions (max = 6):', min_value=1, max_value=6, value=1)

# Inicializar el tablero
def initialize_board():
    return [["" for _ in range(3)] for _ in range(3)]

# Función para seleccionar un espacio
def select_space(board, move, turn):
    row = (move-1) // 3
    col = (move-1) % 3
    if board[row][col] == "":
        board[row][col] = turn
        return True
    return False

# Verificar si un jugador ha ganado
def has_won(board, player):
    return any(
        all(board[i][j] == player for j in range(3)) for i in range(3)
    ) or any(
        all(board[i][j] == player for i in range(3)) for j in range(3)
    ) or all(
        board[i][i] == player for i in range(3)
    ) or all(
        board[i][2-i] == player for i in range(3)
    )

# Verificar si el juego ha terminado
def game_is_over(board):
    return has_won(board, "X") or has_won(board, "O") or all(cell != "" for row in board for cell in row)

# Dibujar el tablero con Matplotlib
def draw_board(board):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.grid(True, which='both', color='black', linewidth=2)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Dibujar X y O
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                ax.plot(j, 2 - i, 'bo', markersize=30)  # "X" en azul
            elif board[i][j] == "O":
                ax.plot(j, 2 - i, 'ro', markersize=30)  # "O" en rojo
    
    st.pyplot(fig)

# Encontrar el mejor movimiento de la IA
def find_best_move(board, depth):
    best_move = None
    best_value = float('inf')
    for move in available_moves(board):
        new_board = deepcopy(board)
        select_space(new_board, move, "O")
        move_value = minimax_alphabeta(new_board, depth, -float('inf'), float('inf'), True)
        if move_value < best_value:
            best_value = move_value
            best_move = move
    return best_move

# Función para obtener movimientos disponibles
def available_moves(board):
    return [i * 3 + j + 1 for i in range(3) for j in range(3) if board[i][j] == ""]

# Función minimax con poda alfa-beta
def minimax_alphabeta(board, depth, alpha, beta, is_maximizing):
    if game_is_over(board) or depth == 0:
        return evaluate_board(board)
    if is_maximizing:
        max_eval = -float('inf')
        for move in available_moves(board):
            new_board = deepcopy(board)
            select_space(new_board, move, "X")
            eval = minimax_alphabeta(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in available_moves(board):
            new_board = deepcopy(board)
            select_space(new_board, move, "O")
            eval = minimax_alphabeta(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Evaluar el tablero
def evaluate_board(board):
    if has_won(board, "X"):
        return 1
    elif has_won(board, "O"):
        return -1
    else:
        return 0

# Verificar si el juego ha terminado
if game_is_over(st.session_state.board):
    if has_won(st.session_state.board, "X"):
        st.session_state.winner = "Player (X)"
    elif has_won(st.session_state.board, "O"):
        st.session_state.winner = "IA (O)"
    else:
        st.session_state.winner = "Tie"

# Dibujar el tablero
draw_board(st.session_state.board)

# Botones para el jugador
for i in range(1, 10):
    if st.button(f"Place {i}", key=f"{i}"):
        if st.session_state.winner is None:
            # Turno del jugador
            if select_space(st.session_state.board, i, "X"):
                if not game_is_over(st.session_state.board):
                    # Turno de la IA
                    best_move = find_best_move(st.session_state.board, levelOfFkup)
                    select_space(st.session_state.board, best_move, "O")

# Mostrar resultado
if st.session_state.winner:
    st.subheader(f"Result: {st.session_state.winner}")

# Botón para reiniciar el juego
if st.button("Restart Game"):
    st.session_state.board = initialize_board()
    st.session_state.winner = None
