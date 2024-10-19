import streamlit as st
import numpy as np

# Inicializar el tablero vacío
def inicializar_tablero():
    return np.full((3, 3), "")

# Verificar si hay un ganador o si el juego es un empate
def verificar_ganador(tablero):
    # Verificar filas, columnas y diagonales
    for i in range(3):
        if tablero[i, 0] == tablero[i, 1] == tablero[i, 2] != "":
            return tablero[i, 0]
        if tablero[0, i] == tablero[1, i] == tablero[2, i] != "":
            return tablero[0, i]
    
    if tablero[0, 0] == tablero[1, 1] == tablero[2, 2] != "":
        return tablero[0, 0]
    if tablero[0, 2] == tablero[1, 1] == tablero[2, 0] != "":
        return tablero[0, 2]

    # Verificar empate
    if not np.any(tablero == ""):
        return "Empate"
    
    return None

# Implementación del algoritmo Minimax con poda alfa-beta
def minimax(tablero, profundidad, es_maximizador, limite_profundidad, alpha, beta):
    ganador = verificar_ganador(tablero)
    if ganador == "X":
        return -10 + profundidad
    elif ganador == "O":
        return 10 - profundidad
    elif ganador == "Empate":
        return 0

    # Limitar la profundidad del árbol
    if profundidad >= limite_profundidad:
        return 0

    if es_maximizador:
        mejor_valor = -np.inf
        for i in range(3):
            for j in range(3):
                if tablero[i, j] == "":
                    tablero[i, j] = "O"
                    valor = minimax(tablero, profundidad + 1, False, limite_profundidad, alpha, beta)
                    tablero[i, j] = ""
                    mejor_valor = max(mejor_valor, valor)
                    alpha = max(alpha, valor)
                    if beta <= alpha:
                        break  # Poda beta
        return mejor_valor
    else:
        mejor_valor = np.inf
        for i in range(3):
            for j in range(3):
                if tablero[i, j] == "":
                    tablero[i, j] = "X"
                    valor = minimax(tablero, profundidad + 1, True, limite_profundidad, alpha, beta)
                    tablero[i, j] = ""
                    mejor_valor = min(mejor_valor, valor)
                    beta = min(beta, valor)
                    if beta <= alpha:
                        break  # Poda alfa
        return mejor_valor

# IA elige el mejor movimiento usando minimax con poda alfa-beta
def mejor_movimiento(tablero, limite_profundidad):
    mejor_valor = -np.inf
    movimiento = None
    alpha = -np.inf
    beta = np.inf
    for i in range(3):
        for j in range(3):
            if tablero[i, j] == "":
                tablero[i, j] = "O"
                valor = minimax(tablero, 0, False, limite_profundidad, alpha, beta)
                tablero[i, j] = ""
                if valor > mejor_valor:
                    mejor_valor = valor
                    movimiento = (i, j)
                alpha = max(alpha, valor)
    return movimiento

# Inicializar el tablero en Streamlit
if "tablero" not in st.session_state:
    st.session_state.tablero = inicializar_tablero()
    st.session_state.turno = "X"  # El jugador humano empieza
    st.session_state.mensaje = "Tu turno (X)"
    st.session_state.dificultad = 3  # Dificultad por defecto "Medio"

# Mostrar el título y el subheader
st.set_page_config(page_title="Tic-Tac-Toe with Minimax and Alpha-Beta Pruning", page_icon="💻", layout="wide")
st.title("Tic-Tac-Toe with Minimax and Alpha-Beta Pruning")
st.subheader("Made by: Arian Vazquez")

# Mensaje explicativo para el usuario
st.write("**Note:** You may need to click the buttons twice for the movement to register or to restart the game.")

# Selección de la dificultad
dificultad = st.selectbox("Select the difficulty:", ["Easy", "Medium", "Hard"])

# Establecer la profundidad según la dificultad seleccionada
if dificultad == "Easy":
    limite_profundidad = 1
elif dificultad == "Medium":
    limite_profundidad = 3
else:
    limite_profundidad = 6

st.session_state.dificultad = limite_profundidad

# Mostrar el tablero
st.write(st.session_state.mensaje)
tablero = st.session_state.tablero

# Crear botones para el tablero
ganador = verificar_ganador(tablero)  # Verificar ganador antes de actualizar los botones
for i in range(3):
    cols = st.columns(3)
    for j in range(3):
        if tablero[i, j] == "" and not ganador:
            if cols[j].button(" ", key=f"{i}-{j}"):
                if st.session_state.turno == "X":
                    tablero[i, j] = "X"
                    st.session_state.turno = "O"
                    ganador = verificar_ganador(tablero)
                    if ganador:
                        st.session_state.mensaje = f"Ganador: {ganador}"
                    else:
                        st.session_state.mensaje = "Turno de la IA (O)"
                    st.rerun()
        else:
            cols[j].button(tablero[i, j], key=f"{i}-{j}", disabled=True)

# Turno de la IA: Si hay más "X" que "O", la IA realiza su movimiento
if not ganador and np.sum(tablero == "X") > np.sum(tablero == "O"):
    movimiento = mejor_movimiento(tablero, st.session_state.dificultad)
    if movimiento:
        tablero[movimiento] = "O"
        st.session_state.turno = "X"
        ganador = verificar_ganador(tablero)
        if ganador:
            st.session_state.mensaje = f"Ganador: {ganador}"
        else:
            st.session_state.mensaje = "Tu turno (X)"
        st.rerun()

# Reiniciar el juego
if st.button("Reiniciar"):
    st.session_state.tablero = inicializar_tablero()
    st.session_state.turno = "X"
    st.session_state.mensaje = "Tu turno (X)"
    st.rerun()
