# Artificial Intelligence Resources New (About this repository)
Welcome to the Artificial-Intelligence-Resources-New repository! This repository is dedicated to providing a comprehensive collection of resources related to Artificial Intelligence (AI) and the databases crucial for AI research and development. Whether you are a beginner looking to dive into the world of AI or an experienced practitioner seeking advanced datasets, this repository aims to be a valuable resource for you.

# Logic Gate Neurons: Learning in Action
In this project, I used Python to create a simple artificial neural network, focusing on perceptrons, to simulate basic logic gates such as AND, OR, and XOR. The perceptron is the simplest type of neural network, with a single layer of neurons that perform binary classification based on a linear decision boundary.

The project aimed to demonstrate how perceptrons can be trained to mimic the behavior of these gates by adjusting their weights and bias through a supervised learning algorithm. For each gate, I defined input and expected output pairs, then trained the perceptron to learn the correct mapping. The perceptron successfully learned the AND, OR gates, showcasing its ability to model linearly separable problems.

However, a key part of the project was exploring the limitations of the perceptron. While it can easily simulate simple logic gates, it fails to model the XOR gate due to the non-linearly separable nature of its output. This led to a discussion on the need for multi-layer networks (such as multi-layer perceptrons or deep neural networks) to handle more complex decision boundaries.

# Adaptive Rock Paper Scissors Game against AI using Deep Q-Learning
This project focuses on building an intelligent agent capable of playing and adapting to the classic game of Rock Paper Scissors in real-time using Deep Q-Learning, a reinforcement learning technique. The goal was to create an AI that could not only play against a human opponent but also learn and adjust its strategy based on the player's behavior over time.

In the game, the AI initially plays randomly, as it has no prior knowledge of the player's patterns. However, as the game progresses, the AI gathers data from previous rounds, learning the tendencies and strategies of the opponent. Using Deep Q-Learning, the AI updates its Q-values (a table that helps the agent decide which action to take) by rewarding successful predictions and penalizing wrong moves. Over time, the AI improves its decision-making by identifying patterns in the player's choices and adapting its strategy accordingly.

The neural network used in the Deep Q-Learning model helps generalize from previous game experiences, making the AI more efficient in selecting moves that counter the player's behavior. The result is an adaptive and competitive opponent that grows more challenging the longer the game is played.

This project demonstrates the potential of reinforcement learning techniques in real-time decision-making environments. It highlights how AI can adapt to human behavior dynamically, offering a fun and interactive way to explore advanced AI concepts.

# AI to predict feelings from text
In this project, I developed an AI model to detect six core emotions—Sadness, Joy, Love, Anger, Fear, and Surprise—from textual data using Natural Language Processing (NLP) techniques. The goal was to classify a given piece of text into one of these emotional categories by training specialized models for each emotion using a One-vs-Rest (OVR) multi-class classification approach.

The core of the project involved building a separate AI classifier for each emotion. Using Hugging Face’s emotional datasets, I fine-tuned each AI to specialize in recognizing a particular emotion. The OVR strategy allows each model to distinguish its respective emotion from all other possibilities, making the system highly accurate in determining the most likely emotional response for a given text.

By leveraging state-of-the-art NLP techniques and transfer learning, the AI processed text input, analyzed it for linguistic cues, and predicted which of the six emotions was most strongly conveyed. This method enabled more precise emotion detection, as each model was optimized to focus on subtle patterns associated with its specific emotion.

The project demonstrates the effectiveness of using multi-class classification with specialized models for complex tasks like emotion detection, offering potential applications in sentiment analysis, mental health monitoring, and customer feedback analysis.

# Maze Generator and Solver (using A*)
In this project, I developed a maze generation and solving application that employs the A* search algorithm to find optimal paths within generated mazes. The primary objective was to showcase the application of pathfinding algorithms in a gaming context while providing users with an interactive experience.

The application begins by dynamically generating a maze using a randomized algorithm, ensuring that each maze is unique and presents different challenges. Once the maze is created, the user has the option to visualize the maze-solving process. By implementing the A* algorithm, the program intelligently explores potential paths from the start to the goal point, considering both the cost of the path and heuristic estimates of distance to the goal.

As the algorithm processes, it displays the exploration of nodes, allowing users to see how the A* algorithm evaluates and chooses the best path. When the solution is found, the program highlights the route taken, providing a clear visual representation of the optimal path through the maze.

This project effectively illustrates the principles of search algorithms and their applications in game design, enhancing user engagement through visualization and interaction. It also serves as a foundation for exploring more complex pathfinding techniques and maze generation algorithms.

# Customer Segmentation Tool with K-means
In this project, I developed a Customer Segmentation Tool utilizing the K-means clustering algorithm to analyze and categorize customers based on their purchasing behavior. The primary goal was to demonstrate how unsupervised learning techniques can simplify complex problems and uncover valuable insights from customer data.

The project involved preprocessing customer data, which included features such as age, income, and spending habits. By applying the K-means algorithm, I was able to segment customers into distinct clusters, each representing different behavioral patterns. This segmentation helps businesses tailor their marketing strategies, improve customer engagement, and enhance overall customer satisfaction.

To provide a comprehensive understanding of the results, I created two detailed reports (in English and Spanish) that outline the analysis, methodology, and insights derived from the segmentation process. These reports include visualizations of the clusters, highlighting the characteristics of each customer segment, making it easier for stakeholders to interpret the findings.

This project effectively illustrates the power of K-means clustering in customer analysis and demonstrates how unsupervised models can simplify complex datasets into actionable insights, enabling businesses to make data-driven decisions.

# Generador de Sudoku
In this project, I developed a Sudoku Generator that creates fully functional Sudoku puzzles of varying difficulty levels. The primary goal was to showcase algorithmic problem-solving techniques and provide users with a fun and engaging way to enjoy this classic game.

The generator utilizes a backtracking algorithm, a depth-first search technique that systematically fills the Sudoku grid while adhering to the game's rules. This ensures that each generated puzzle has a unique solution. After creating the complete Sudoku grid, the generator removes a predetermined number of cells based on the desired difficulty level, creating a challenging yet solvable puzzle for players.

To enhance user experience, the application features an intuitive interface that allows users to select the difficulty level and generate new puzzles at the click of a button. Additionally, users can check their solutions against the correct answers, ensuring an enjoyable and educational experience as they work through the puzzles.

This project effectively demonstrates the principles of algorithm design and implementation, while providing a practical application of programming concepts in creating interactive games.

# Titanic survival algorithm
In this project, I developed a predictive model to analyze the survival rates of passengers on the Titanic using Logistic Regression, a powerful statistical method for binary classification. The primary aim was to demonstrate the algorithm's capacity for categorization by predicting whether a passenger survived or not based on various features.

The analysis began with a comprehensive exploration of the Titanic dataset, which includes attributes such as passenger age, gender, ticket class, and data about their social relationships. After preprocessing the data to handle missing values and categorical variables, I employed the Logistic Regression model to train the algorithm. The model learns to identify patterns and relationships within the dataset, allowing it to make informed predictions on survival outcomes.

This project effectively illustrates the practical application of Logistic Regression in real-world scenarios, showcasing the power of machine learning algorithms in classification tasks and their potential impact in decision-making processes.
