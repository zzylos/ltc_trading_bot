import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os
from config.settings import RL_CONFIG
import logging

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # Buy, Sell, Hold
        
        # Hyperparameters
        self.gamma = RL_CONFIG['discount_factor']  # Discount factor
        self.epsilon = RL_CONFIG['exploration_rate']  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = RL_CONFIG['learning_rate']
        self.batch_size = RL_CONFIG['batch_size']
        self.memory_size = RL_CONFIG['memory_size']
        self.update_frequency = RL_CONFIG['update_frequency']
        
        # Replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Build main and target networks
        self.main_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Counter for update frequency
        self.update_counter = 0
        
        # Logger
        self.logger = logging.getLogger('DQNAgent')
        
        # Create directory for saved models
        self.model_dir = RL_CONFIG['model_save_path']
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _build_model(self):
        """Build a neural network for the DQN agent"""
        model = Sequential([
            LSTM(64, input_shape=(self.state_size[0], self.state_size[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def update_target_network(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.main_model.get_weights())
        self.logger.info("Target network updated")
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose an action based on the current state"""
        # Exploration: choose a random action
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action based on model prediction
        act_values = self.main_model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the model with random samples from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:
                # Q-Learning formula: Q(s,a) = r + γ * max(Q'(s',a'))
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            
            # Update target for the specific action
            target_f = self.main_model.predict(state, verbose=0)
            target_f[0][action] = target
            
            # Train the model
            self.main_model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay epsilon for less exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update counter and potentially update target network
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self.update_target_network()
            self.update_counter = 0
    
    def load(self, name):
        """Load a saved model"""
        try:
            self.main_model.load_weights(name)
            self.target_model.load_weights(name)
            self.logger.info(f"Model loaded from: {name}")
        except:
            self.logger.error(f"Failed to load model from: {name}")
    
    def save(self, name):
        """Save the model"""
        try:
            self.main_model.save_weights(name)
            self.logger.info(f"Model saved to: {name}")
        except:
            self.logger.error(f"Failed to save model to: {name}")