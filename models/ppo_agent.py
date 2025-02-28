import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from config.settings import RL_CONFIG
import logging
import os

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = RL_CONFIG['discount_factor']
        self.clip_ratio = 0.2  # PPO clip ratio
        self.policy_update_epochs = 4
        self.value_update_epochs = 4
        self.batch_size = RL_CONFIG['batch_size']
        self.learning_rate = RL_CONFIG['learning_rate']
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Tracking variables
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        
        # Logger
        self.logger = logging.getLogger('PPOAgent')
        
        # Create directory for saved models
        self.model_dir = RL_CONFIG['model_save_path']
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _build_actor(self):
        """Build the actor network for policy"""
        inputs = Input(shape=self.state_size)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        
        # Actions probability output (softmax)
        action_probs = Dense(self.action_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=action_probs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _build_critic(self):
        """Build the critic network for value estimation"""
        inputs = Input(shape=self.state_size)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        
        # Value output (single value)
        value = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=value)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def act(self, state, training=True):
        """Choose an action based on the current policy"""
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        if training:
            # Sample from probability distribution
            action = np.random.choice(self.action_size, p=action_probs)
            log_prob = np.log(action_probs[action])
            value = self.critic.predict(state, verbose=0)[0, 0]
            
            return action, log_prob, value
        else:
            # For evaluation/deployment, choose the most probable action
            return np.argmax(action_probs)
    
    def remember(self, state, action, reward, value, log_prob, done):
        """Store experience for later training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def _calculate_advantages(self, rewards, values, dones, next_value):
        """Calculate advantages using Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
        last_value = next_value
        
        # Calculate advantages in reverse order
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            advantages[t] = delta + self.gamma * 0.95 * last_advantage * mask
            
            last_advantage = advantages[t]
            last_value = values[t]
            
        # Calculate returns (value targets)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def train(self, next_state=None, next_value=None):
        """Train the agent on collected experiences"""
        if not self.states:  # No experiences collected
            return
        
        # If no next_value is provided, use 0 for terminal state
        if next_value is None:
            next_value = 0
            
        # Convert lists to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        old_log_probs = np.array(self.log_probs)
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_advantages(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # PPO Update Loop
        indices = np.arange(len(states))
        
        for _ in range(self.policy_update_epochs):
            # Shuffle for stochastic gradient descent
            np.random.shuffle(indices)
            
            # Process in batches
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Training data for this batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor (Policy) Update
                with tf.GradientTape() as tape:
                    # Get new action probabilities
                    new_action_probs = self.actor(batch_states, training=True)
                    
                    # Select probabilities of the actions that were taken
                    batch_indices_range = np.arange(len(batch_actions))
                    selected_probs = tf.gather_nd(new_action_probs, 
                                                 list(zip(batch_indices_range, batch_actions)))
                    
                    # Calculate new log probabilities
                    new_log_probs = tf.math.log(selected_probs + 1e-10)
                    
                    # Calculate ratio (new_prob / old_prob)
                    ratios = tf.exp(new_log_probs - batch_old_log_probs)
                    
                    # Calculate PPO loss components
                    surrogate1 = ratios * batch_advantages
                    surrogate2 = tf.clip_by_value(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages
                    
                    # Policy loss (negative because we want to maximize)
                    policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Add entropy bonus for exploration
                    entropy = -tf.reduce_sum(new_action_probs * tf.math.log(new_action_probs + 1e-10), axis=1)
                    entropy_bonus = 0.01 * tf.reduce_mean(entropy)
                    
                    total_loss = policy_loss - entropy_bonus
                
                # Get gradients and apply to actor network
                policy_grads = tape.gradient(total_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))
                
                # Critic (Value) Update
                for _ in range(self.value_update_epochs):
                    self.critic.fit(
                        batch_states, 
                        batch_returns, 
                        batch_size=len(batch_states), 
                        verbose=0,
                        shuffle=True
                    )
        
        # Clear memory after optimization is complete
        self._clear_memory()
        
        self.logger.info("PPO model trained")
    
    def _clear_memory(self):
        """Clear the agent's memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
    
    def load(self, actor_path, critic_path):
        """Load saved models"""
        try:
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            self.logger.info(f"Models loaded from: {actor_path} and {critic_path}")
        except:
            self.logger.error(f"Failed to load models")
    
    def save(self, actor_path, critic_path):
        """Save models"""
        try:
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            self.logger.info(f"Models saved to: {actor_path} and {critic_path}")
        except:
            self.logger.error(f"Failed to save models")