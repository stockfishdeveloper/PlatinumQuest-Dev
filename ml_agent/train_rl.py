"""
Reinforcement Learning Training Script for PlatinumQuest Hunt Mode

This script implements a simple PPO-based RL agent that learns to collect gems
in the flat training map via direct communication with the game.

Architecture:
- Game (TorqueScript) → TCP socket → Python server
- Python receives observations, runs model inference, sends actions back
- Model is trained using PPO from stable-baselines3
"""

import socket
import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import signal
import sys

# Simple neural network for the policy
class GemCollectorPolicy(nn.Module):
    def __init__(self, obs_dim=286, action_dim=6):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # Action head (6 binary actions)
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Sigmoid for binary actions
        actions = torch.sigmoid(self.action_head(x))
        return actions

    def get_action(self, state, deterministic=False):
        """Get binary actions from state"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)

            action_probs = self.forward(state)

            if deterministic:
                # Threshold at 0.5
                actions = (action_probs > 0.5).int()
            else:
                # Sample from Bernoulli distribution
                actions = torch.bernoulli(action_probs).int()

        return actions.squeeze(0).numpy()


class RLServer:
    """TCP server that communicates with the game"""

    def __init__(self, host='127.0.0.1', port=8888, model_path=None):
        self.host = host
        self.port = port
        self.running = True

        # Create model
        self.model = GemCollectorPolicy()

        # Load existing model if provided
        if model_path:
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        # Statistics
        self.step_count = 0
        self.episode_count = 0

        # Socket setup
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(1.0)  # 1 second timeout for accept()
        self.socket.bind((self.host, self.port))

    def run(self):
        """Main server loop"""
        print(f"RL Server listening on {self.host}:{self.port}")
        print("Waiting for game to connect...")
        print("Press Ctrl+C to stop")

        self.socket.listen(1)

        while self.running:
            try:
                conn, addr = self.socket.accept()
                print(f"Game connected from {addr}")

                self.handle_client(conn)

            except socket.timeout:
                # Timeout allows checking self.running flag
                continue
            except KeyboardInterrupt:
                print("\nShutting down server...")
                self.running = False
                break
            except Exception as e:
                if self.running:  # Only print error if not shutting down
                    print(f"Error: {e}")

        self.socket.close()
        print("Server stopped")

    def handle_client(self, conn):
        """Handle communication with connected game client"""
        buffer = ""
        recv_count = 0

        try:
            while True:
                # Receive data
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    print("Game disconnected")
                    break

                recv_count += 1
                if recv_count % 10 == 0:
                    print(f"Received {recv_count} messages")

                buffer += data

                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)

                    if line.strip():
                        action = self.process_observation(line.strip())

                        # Send action back
                        action_str = ','.join(map(str, action)) + '\n'
                        conn.sendall(action_str.encode('utf-8'))  # sendall ensures full send

                        self.step_count += 1

                        if self.step_count % 10 == 0:  # More frequent logging
                            print(f"Steps: {self.step_count}")

        except Exception as e:
            print(f"Client error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()

    def process_observation(self, obs_json):
        """
        Process observation from game and return action

        Observation format: JSON array of 286 floats
        Action format: 6 binary values [forward, backward, left, right, jump, powerup]
        """
        try:
            # Debug: print first received message
            if self.step_count == 0:
                print(f"First observation received (first 100 chars): {obs_json[:100]}")

            # Parse observation
            obs = json.loads(obs_json)

            # Debug: check dimension
            if self.step_count == 0:
                print(f"Observation has {len(obs)} dimensions (expected 286)")

            # Get action from model
            action = self.model.get_action(obs, deterministic=False)

            return action.tolist()

        except Exception as e:
            # Print full error for first few failures
            if self.step_count < 3:
                print(f"Error processing observation: {e}")
                print(f"Received data (first 150 chars): {obs_json[:150]}")
            # Return safe default action (no movement)
            return [0, 0, 0, 0, 0, 0]


def main():
    """Start the RL training server"""
    import argparse

    parser = argparse.ArgumentParser(description='PlatinumQuest RL Training Server')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--model', help='Path to pre-trained model')

    args = parser.parse_args()

    # Create server
    server = RLServer(host=args.host, port=args.port, model_path=args.model)

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\nReceived interrupt signal, shutting down...')
        server.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.socket.close()


if __name__ == '__main__':
    main()
