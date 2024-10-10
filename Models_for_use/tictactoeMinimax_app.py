import streamlit as st
from copy import deepcopy

st.set_page_config(page_title="Tic-Tac-Toe game against an AI", page_icon="🎮")

# Interfaz de usuario
st.title("Tic-Tac-Toe game against an AI 🤖")

st.subheader("Made by: Arian Vazquez Fernandez")

# Example data
st.header("Understanding the model")

st.markdown("""
- Depth of the decision tree: The depth of the decision tree is the is the prediction capacity of the model to analyze movements made by the player and artificial intelligence.
- What happens in the depth 4?: Analyzing the future is a great power but you must know how to interpret that data, the model sees a future possibility in the game to win, but it cannot see other better possibilities in the nearby game.
- ⚠️ If you are playing on a phone, play with the screen horizontal and not vertical to place the spaces correctly. ⚠️
""")

# Example data
st.header("Let's use the model")

levelOfFkup = st.number_input('Depth of the decision tree that AI predicts (max = 6):', min_value=1, max_value=6, value=1)

st.write("Place your token on one of the buttons:")

# Función para inicializar el tablero
def initialize_board():
    return [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]

# Función para seleccionar un espacio
def select_space(board, move, turn):
    if move not in range(1, 10):
        return False
    row = (move-1) // 3
    col = (move-1) % 3
    if board[row][col] != "X" and board[row][col] != "O":
        board[row][col] = turn
        return True
    else:
        return False

# Función para obtener los movimientos disponibles
def available_moves(board):
    moves = []
    for row in board:
        for col in row:
            if col != "X" and col != "O":
                moves.append(int(col))
    return moves

# Función para verificar si un jugador ha ganado
def has_won(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    for i in range(3):
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False

# Función para verificar si el juego ha terminado
def game_is_over(board):
    return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

# Función para evaluar el tablero
def evaluate_board(board):
    if has_won(board, "X"):
        return 1
    elif has_won(board, "O"):
        return -1
    else:
        return 0

# Función para ordenar los movimientos
def sort_moves(board, moves, is_maximizing):
    move_values = []
    for move in moves:
        new_board = deepcopy(board)
        select_space(new_board, move, "X" if is_maximizing else "O")
        move_values.append((move, evaluate_board(new_board)))
    move_values.sort(key=lambda x: x[1], reverse=is_maximizing)
    return [move for move, value in move_values]

# Función de minimax con poda alfa-beta
def minimax_alphabeta(board, depth, alpha, beta, is_maximizing):
    if game_is_over(board) or depth == 0:
        return evaluate_board(board)
    if is_maximizing:
        max_eval = -float('inf')
        for move in sort_moves(board, available_moves(board), is_maximizing):
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
        for move in sort_moves(board, available_moves(board), is_maximizing):
            new_board = deepcopy(board)
            select_space(new_board, move, "O")
            eval = minimax_alphabeta(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Función para encontrar el mejor movimiento
def find_best_move(board, is_maximizing, depth):
    best_move = -1
    best_value = -float('inf') if is_maximizing else float('inf')
    for move in available_moves(board):
        new_board = deepcopy(board)
        select_space(new_board, move, "X" if is_maximizing else "O")
        board_value = minimax_alphabeta(new_board, depth, -float('inf'), float('inf'), not is_maximizing)
        if is_maximizing:
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
    return best_move

# Inicializar el tablero
if 'board' not in st.session_state:
    st.session_state.board = initialize_board()
    st.session_state.current_player = 'X'
    st.session_state.winner = None

# Mostrar el tablero
def display_board(board):
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            cell = board[i][j]
            button_label = cell if cell in ["X", "O"] else ""
            if cols[j].button(button_label, key=f"{i}-{j}"):
                if st.session_state.winner is None and cell not in ["X", "O"]:
                    # Jugador coloca su "X"
                    if select_space(st.session_state.board, int(cell), "X"):
                        st.experimental_rerun()  # Volvemos a renderizar la interfaz tras mover la "X"

                        # Verificar si el juego terminó tras el movimiento del jugador
                        if not game_is_over(st.session_state.board):
                            # IA coloca su "O"
                            best_move = find_best_move(st.session_state.board, False, levelOfFkup)
                            select_space(st.session_state.board, best_move, "O")
                            st.experimental_rerun()  # Volvemos a renderizar tras mover la "O"

                        # Verificar si alguien ha ganado después del turno de la IA
                        if game_is_over(st.session_state.board):
                            if has_won(st.session_state.board, "X"):
                                st.session_state.winner = "Player (X)"
                            elif has_won(st.session_state.board, "O"):
                                st.session_state.winner = "IA (O)"
                            else:
                                st.session_state.winner = "Tie"
                            st.experimental_rerun()

# Verificar el estado del juego
if game_is_over(st.session_state.board):
    if has_won(st.session_state.board, "X"):
        st.session_state.winner = "Player (X)"
    elif has_won(st.session_state.board, "O"):
        st.session_state.winner = "IA (O)"
    else:
        st.session_state.winner = "Tie"

display_board(st.session_state.board)

if st.session_state.winner:
    st.subheader(f"Result: {st.session_state.winner}")

# Botón para reiniciar el juego
if st.button("Restart Game"):
    st.session_state.board = initialize_board()
    st.session_state.current_player = 'X'
    st.session_state.winner = None
    st.experimental_rerun()
