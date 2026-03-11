# Self-Driving Car Neural Evolution Simulator 🚗🧠

A from-scratch implementation of a **self-driving car training simulator** using **Neural Networks and Evolutionary Algorithms**.

This project was inspired by concepts introduced in Harvard's CS50 AI course and recreates the idea completely from scratch using only essential Python libraries.

The simulator evolves generations of cars that gradually learn how to navigate a track and reach the goal through **selection, mutation, and survival of the fittest**.

---

## Demo

The simulator trains multiple cars simultaneously.

Each generation:
- Cars sense the environment
- Neural networks decide steering actions
- Poor performers are eliminated
- The best cars reproduce with mutation

Over time, the population learns to successfully complete the track.

---

## Features

- Evolutionary neural network training
- Population-based learning
- Real-time simulation
- Track sensing system
- Fitness-based selection
- Mutation and crossover
- Visualization of training progress

Optional additions:

- Video recording of the evolution process
- Alive car counter
- Generation tracking

---

## Project Structure


self-driving-evolution/
│
├── main.py # Entry point
├── config.py # Simulation configuration
├── requirements.txt # Dependencies
│
├── cars/ # Car agent logic
│ ├── car.py
│ └── sensors.py
│
├── neural_network/ # Neural network implementation
│ └── network.py
│
├── evolution/ # Genetic algorithm logic
│ ├── population.py
│ ├── mutation.py
│ └── selection.py
│
├── simulation/ # Simulation environment
│ ├── track.py
│ └── physics.py
│
├── visualization/ # Rendering and UI
│ └── renderer.py
│
└── assets/ # Track images and assets


---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/self-driving-evolution.git
cd self-driving-evolution

Create a virtual environment:

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

## Install dependencies:

pip install -r requirements.txt
Running the Simulator
python main.py

You will see a window where:

Cars spawn

Neural networks control steering

Evolution occurs generation by generation

Eventually one of the cars learns to complete the track.

## How the AI Works

#### Neural Network

Each car has a small feed-forward neural network that receives sensor inputs:

distance_left
distance_front
distance_right
velocity
angle

Outputs:

steering_left
steering_right
accelerate
Evolutionary Training

Each generation:

Cars are evaluated by fitness

Top performers survive

New population is created via:

mutation

crossover

Training repeats

Over many generations the population improves.

Technologies Used

Python

NumPy

OpenCV (optional recording)

Matplotlib / PyGame (visualization)

No heavy ML frameworks were used.

The neural networks and evolution logic were implemented from scratch.

Inspiration

Inspired by the concepts taught in:

Harvard CS50's Introduction to Artificial Intelligence with Python

Future Improvements

Possible upgrades:

Reinforcement learning hybrid models

GPU acceleration

Web-based visualization

Advanced track environments

Multi-agent competitions

## License

Apache License

Author

Built by Willy

Machine Learning Engineer | AI Developer | Blockchain Developer
