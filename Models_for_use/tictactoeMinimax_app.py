import streamlit as st
from copy import deepcopy

st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮")

# Interfaz de usuario
st.title("Tic-Tac-Toe game against an AI 🤖")

st.subheader("Made by: Arian Vazquez Fernandez")

# Explicación del modelo
st.header("Understanding the model")

st.markdown("""
- Depth of the decision tree: The depth of the decision tree is the prediction capacity of the model to analyze movements made by the player and artificial intelligence.
- What happens in depth 4?: Analyzing future moves can help the AI find a path to victory, but it might miss better moves in the nearby game.
- ⚠️ If you're playing on a phone, turn your screen horizontal for better spacing. ⚠️
""")

# Selección del nivel de dificultad
levelOfFkup = st.number_input('Depth of the decision tree for AI predictions (max = 6):', min_value=1, max_value=6, value=1)

st.write("Place your token on one of the buttons:")

# Función para inicializar el tablero
def initialize_board():
    return [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]

# Función para evaluar el tablero
def evaluate_board(board):
    if has_won(board, "X"):
        return 1
    elif has_won(board, "O"):
        return -1
    else:
        return 0

# Función para seleccionar un espacio en el tablero
def select_space(board, move, turn):
    row = (move-1) // 3
    col = (move-1) % 3
    if board[row][col] != "X" and board[row][col] != "O":
        board[row][col] = turn
        return True
    return False

# Función para obtener movimientos disponibles
def available_moves(board):
    return [int(col) for row in board for col in row if col not in ["X", "O"]]

# Función para verificar si un jugador ha ganado
def has_won(board, player):
    return any(
        all(board[i][j] == player for j in range(3)) for i in range(3)  # filas
    ) or any(
        all(board[i][j] == player for i in range(3)) for j in range(3)  # columnas
    ) or all(
        board[i][i] == player for i in range(3)  # diagonal principal
    ) or all(
        board[i][2-i] == player for i in range(3)  # diagonal secundaria
    )

# Verificar si el juego ha terminado
def game_is_over(board):
    return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

# Función de minimax con poda alfa-beta
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

# Función para encontrar el mejor movimiento de la IA
def find_best_move(board, depth):
    best_move = None
    best_value = float('inf')  # IA es minimizante (O)
    for move in available_moves(board):
        new_board = deepcopy(board)
        select_space(new_board, move, "O")
        move_value = minimax_alphabeta(new_board, depth, -float('inf'), float('inf'), True)
        if move_value < best_value:
            best_value = move_value
            best_move = move
    return best_move

# Inicializar el tablero
if 'board' not in st.session_state:
    st.session_state.board = initialize_board()
    st.session_state.winner = None

# Mostrar el tablero con etiquetas en los botones
def display_board(board):
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            cell = board[i][j]
            button_label = cell if cell in ["X", "O"] else ""
            if cols[j].button(button_label, key=f"{i}-{j}"):
                if st.session_state.winner is None and cell not in ["X", "O"]:
                    # El jugador hace su movimiento
                    if select_space(st.session_state.board, int(cell), "X"):
                        # Si el juego no ha terminado, la IA hace su movimiento
                        if not game_is_over(st.session_state.board):
                            best_move = find_best_move(st.session_state.board, levelOfFkup)
                            select_space(st.session_state.board, best_move, "O")
                        # Verifica si alguien ganó
                        if game_is_over(st.session_state.board):
                            if has_won(st.session_state.board, "X"):
                                st.session_state.winner = "Player (X)"
                            elif has_won(st.session_state.board, "O"):
                                st.session_state.winner = "IA (O)"
                            else:
                                st.session_state.winner = "Tie"

# Verificar si el juego ha terminado
if game_is_over(st.session_state.board):
    if has_won(st.session_state.board, "X"):
        st.session_state.winner = "Player (X)"
    elif has_won(st.session_state.board, "O"):
        st.session_state.winner = "IA (O)"
    else:
        st.session_state.winner = "Tie"

# Mostrar el tablero
display_board(st.session_state.board)

# Mostrar el resultado
if st.session_state.winner:
    st.subheader(f"Result: {st.session_state.winner}")

# Botón para reiniciar el juego
if st.button("Restart Game"):
    st.session_state.board = initialize_board()
    st.session_state.winner = None
