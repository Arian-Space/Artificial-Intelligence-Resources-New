import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Create game board
class RPSGame:
    def play(self, player_move, opponent_move):
        moves = ["rock", "paper", "scissors"]
        if moves[player_move] == moves[opponent_move]:
            return 0  # Tie
        elif (moves[player_move] == "rock" and moves[opponent_move] == "scissors") or \
             (moves[player_move] == "paper" and moves[opponent_move] == "rock") or \
             (moves[player_move] == "scissors" and moves[opponent_move] == "paper"):
            return 1  # Player wins
        else:
            return -1  # AI wins

# Create game environment
env = RPSGame()

# Create Q-learning agent
class QLearningAgent:
    def __init__(self, actions, agentName, learning_rate=0.01, discount_factor=0.95, exploration_prob=0.1, memory_size=1000, batch_size=32):
        self.agentName = agentName
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.model = keras.Sequential(name=self.agentName)
        self.model.add(Dense(24, activation='relu', input_shape=(1,), name='Input_layer'))
        self.model.add(Dense(24, activation='relu', name='Hidden_layer'))
        self.model.add(Dense(3, name='output_layer'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

        self.memory = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return random.choice(self.actions)
        else:
            q_values = self.model.predict(np.array([state]))
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            target = reward + self.discount_factor * np.max(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

# Define actions
actions = [0, 1, 2]  # 0: rock, 1: paper, 2: scissors

# Initialize AI agent
@st.cache_resource
def initialize_agent():
    return QLearningAgent(actions, agentName='AdaptiveAI')

# Function to play against AI and train it
def play_and_train(player_move, ai_agent):
    state = random.choice(actions)
    ai_move = ai_agent.choose_action(state)
    result = env.play(player_move, ai_move)
    
    # Train the agent with the new experience
    reward = -result  # From AI's perspective
    next_state = player_move
    ai_agent.remember(state, ai_move, reward, next_state)
    ai_agent.train()
    
    return ai_move, result

# Streamlit page configuration
st.set_page_config(page_title="Adaptive Rock Paper Scissors Game against AI", layout="wide")

st.title("Adaptive Rock Paper Scissors Game against AI")

# Initialize AI agent
ai_agent = initialize_agent()

# Initialize game history in session state
if 'game_history' not in st.session_state:
    st.session_state.game_history = []

# Play against AI
st.header("Play against AI")
player_move = st.radio("Choose your move:", ("Rock", "Paper", "Scissors"))

if st.button("Play"):
    player_move_index = ["rock", "paper", "scissors"].index(player_move.lower())
    ai_move, result = play_and_train(player_move_index, ai_agent)
    
    st.write(f"Your move: {player_move}")
    st.write(f"AI's move: {['Rock', 'Paper', 'Scissors'][ai_move]}")
    
    if result == 1:
        st.write("You won!")
    elif result == -1:
        st.write("You lost.")
    else:
        st.write("It's a tie.")
    
    # Update game history
    st.session_state.game_history.append((player_move_index, ai_move, result))

# Visualize game results
if st.session_state.game_history:
    st.header("Game Results")

    results = [r for _, _, r in st.session_state.game_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(results, bins=[-1.5, -0.5, 0.5, 1.5], color='blue', rwidth=0.8)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['AI Wins', 'Tie', 'Player Wins'])
    ax.set_title('Distribution of Results')
    ax.set_xlabel('Result')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)

    # Show statistics
    total_games = len(results)
    player_wins = results.count(1)
    ai_wins = results.count(-1)
    draws = results.count(0)

    st.write(f"Total games: {total_games}")
    st.write(f"Player wins: {player_wins} ({player_wins/total_games*100:.2f}%)")
    st.write(f"AI wins: {ai_wins} ({ai_wins/total_games*100:.2f}%)")
    st.write(f"Ties: {draws} ({draws/total_games*100:.2f}%)")

# Button to reset the game
if st.button("Reset Game"):
    st.session_state.game_history = []
    ai_agent = initialize_agent()
    st.experimental_rerun()