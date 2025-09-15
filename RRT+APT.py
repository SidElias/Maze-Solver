import turtle
import random
import math
import time
import numpy as np
from collections import defaultdict
import pickle
import os

COLS, ROWS = 20, 20
CELL_SIZE = 25
ANIMATION_DELAY = 0.001
WALL_COLOR = 'black'
START_COLOR = 'green'
END_COLOR = 'red'
PATH_COLOR = 'yellow'
TREE_COLOR = 'blue'
LEARNED_PATH_COLOR = 'orange'

TILES = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

START_IDX = 21
END_IDX = 315

def idx_to_xy(idx):
    return (idx % COLS, idx // COLS)

def xy_to_idx(x, y):
    return y * COLS + x

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3, epsilon_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_history = []
        
    def get_state_features(self, current_pos, goal_pos, nearest_node):
        dist_to_goal = math.hypot(current_pos[0] - goal_pos[0], current_pos[1] - goal_pos[1])
        angle_to_goal = math.atan2(goal_pos[1] - current_pos[1], goal_pos[0] - current_pos[0])
        direction_to_goal = int((angle_to_goal + math.pi) / (math.pi / 4)) % 8
        
        dist_to_nearest = math.hypot(current_pos[0] - nearest_node[0], current_pos[1] - nearest_node[1])
        
        state = (
            min(int(dist_to_goal), 20),
            direction_to_goal,
            min(int(dist_to_nearest), 10)
        )
        return state
    
    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        if next_valid_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            max_next_q = 0
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_q_table = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded_q_table)
            return True
        return False

class Maze:
    def __init__(self, tiles, cols, rows, cell_size, start_idx, end_idx):
        self.tiles = tiles
        self.cols = cols
        self.rows = rows
        self.cell_size = cell_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.cell_centers = {}
        for row in range(self.rows):
            for col in range(self.cols):
                idx = row * self.cols + col
                self.cell_centers[idx] = (col + 0.5, row + 0.5)

    def draw(self, show_visual=True):
        if not show_visual:
            return
            
        turtle.speed(0)
        turtle.hideturtle()
        turtle.tracer(0, 0)
        turtle.bgcolor('white')
        
        for row in range(self.rows):
            for col in range(self.cols):
                idx = row * self.cols + col
                x = col * self.cell_size - (self.cols * self.cell_size) // 2
                y = (self.rows - 1 - row) * self.cell_size - (self.rows * self.cell_size) // 2
                
                if self.tiles[idx] == 0:
                    self._draw_cell(x, y, WALL_COLOR)
                elif idx == self.start_idx:
                    self._draw_cell(x, y, START_COLOR)
                elif idx == self.end_idx:
                    self._draw_cell(x, y, END_COLOR)
        
        turtle.update()

    def _draw_cell(self, x, y, color):
        turtle.up()
        turtle.goto(x, y)
        turtle.down()
        turtle.color('black')
        turtle.fillcolor(color)
        turtle.begin_fill()
        for _ in range(4):
            turtle.forward(self.cell_size)
            turtle.left(90)
        turtle.end_fill()

    def is_free(self, x, y):
        cell_x, cell_y = int(x), int(y)
        if 0 <= cell_x < self.cols and 0 <= cell_y < self.rows:
            idx = xy_to_idx(cell_x, cell_y)
            return self.tiles[idx] != 0
        return False
    
    def grid_to_screen(self, grid_x, grid_y):
        screen_x = grid_x * self.cell_size + self.cell_size // 2 - (self.cols * self.cell_size) // 2
        screen_y = (self.rows - 1 - grid_y) * self.cell_size + self.cell_size // 2 - (self.rows * self.cell_size) // 2
        return screen_x, screen_y
    
    def get_cell_center(self, idx):
        if idx in self.cell_centers:
            return self.cell_centers[idx]
        return idx_to_xy(idx)

class SmartRRTStar:
    def __init__(self, maze, start, goal, agent, step_size=1, max_iterations=1000, animate=True, search_radius=3):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent = agent
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.animate = animate
        self.search_radius = search_radius
        self.nodes = []
        self.parents = {}
        self.costs = {}
        self.node_set = set()
        self.episode_rewards = []
        self.neighbors_map = {}
        
    def get_sampling_actions(self):
        return [
            'random',
            'goal_biased',
            'exploration',
            'path_optimized'
        ]

    def sample_point_with_action(self, action):
        def get_cell_center(x, y):
            return (x + 0.5, y + 0.5)
            
        if action == 'goal_biased':
            if random.random() < 0.3:  # Higher bias towards goal
                # Use exact goal coordinates
                return self.goal
            else:
                # Sample a free cell and return its center
                for _ in range(10):
                    col = random.randint(0, self.maze.cols-1)
                    row = random.randint(0, self.maze.rows-1)
                    if self.maze.is_free(col, row):
                        return get_cell_center(col, row)
                # Fallback if we can't find a free cell
                return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        elif action == 'exploration':
            # Sample points far from existing nodes, but at cell centers
            attempts = 0
            while attempts < 10:
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if not self.maze.is_free(col, row):
                    attempts += 1
                    continue
                    
                center = get_cell_center(col, row)
                min_dist = min([self.dist(center, node) for node in self.nodes])
                if min_dist > 3:  # Far from existing nodes
                    return center
                attempts += 1
                
            # Fallback to any free cell center
            for _ in range(5):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        elif action == 'path_optimized':
            # Sample near the current best path to goal, but at cell centers
            if len(self.nodes) > 5:
                # Pick a random existing node
                node = random.choice(self.nodes[-5:])  # Recent nodes
                
                # Find a free cell near the node
                best_center = None
                min_dist = float('inf')
                
                # Try several cells around the node
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        col = int(node[0]) + dx
                        row = int(node[1]) + dy
                        
                        if 0 <= col < self.maze.cols and 0 <= row < self.maze.rows:
                            if self.maze.is_free(col, row):
                                center = get_cell_center(col, row)
                                d = self.dist(center, self.goal)
                                if d < min_dist:
                                    min_dist = d
                                    best_center = center
                
                if best_center:
                    return best_center
            
            # Fallback to random free cell center
            for _ in range(5):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        else:  # 'random'
            # Random, but still using cell centers
            for _ in range(10):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            
            # Fallback
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))

    def reset(self):
        self.nodes = [self.start]
        self.parents = {self.start: None}
        self.costs = {self.start: 0}
        self.node_set = {self.start}
        self.agent.action_history = []
        self.neighbors_map = {}

    def dist(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_nearest_node(self, random_point):
        return min(self.nodes, key=lambda node: self.dist(node, random_point))
    
    def get_nearby_nodes(self, node, radius):
        return [n for n in self.nodes if self.dist(node, n) <= radius]
        
    def calculate_reward(self, old_node, new_node, action, iteration):
        reward = 0
        
        old_dist = self.dist(old_node, self.goal)
        new_dist = self.dist(new_node, self.goal)
        progress = old_dist - new_dist
        reward += progress * 10
        
        reward -= 0.1
        
        if len(self.nodes) > 1:
            min_dist_to_existing = min([self.dist(new_node, node) for node in self.nodes])
            if min_dist_to_existing > 2:
                reward += 1
        
        if action == 'goal_biased' and progress > 0:
            reward += 2
        elif action == 'exploration' and len(self.nodes) < 50:
            reward += 0.5
        
        if hasattr(self, 'costs') and len(self.costs) > 0:
            nearby_nodes = self.get_nearby_nodes(new_node, self.search_radius)
            potential_costs = []
            for node in nearby_nodes:
                if node in self.costs and not self.check_collision(node, new_node):
                    potential_costs.append(self.costs[node] + self.dist(node, new_node))
            
            if potential_costs:
                min_cost = min(potential_costs)
                if min_cost < old_dist * 2:
                    reward += 2
        
        return reward
        
    def check_collision(self, p1, p2):
        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(p2[0]), float(p2[1])
        
        cell_x0, cell_y0 = int(x0), int(y0)
        cell_x1, cell_y1 = int(x1), int(y1)
        
        if not self.maze.is_free(cell_x0, cell_y0) or not self.maze.is_free(cell_x1, cell_y1):
            return True
            
        dx = cell_x1 - cell_x0
        dy = cell_y1 - cell_y0
        
        if dx == 0 and dy == 0:
            return False
            
        if abs(dx) > 1 or abs(dy) > 1:
            return True
            
        if abs(dx) == 1 and abs(dy) == 1:
            if not self.maze.is_free(cell_x1, cell_y0) or not self.maze.is_free(cell_x0, cell_y1):
                return True
                
        return False
        
    def draw_edge(self, p1, p2, color=TREE_COLOR, width=2):
        if not self.animate:
            return
            
        screen_p1 = self.maze.grid_to_screen(p1[0], p1[1])
        screen_p2 = self.maze.grid_to_screen(p2[0], p2[1])
        
        turtle.up()
        turtle.goto(screen_p1[0], screen_p1[1])
        turtle.down()
        turtle.color(color)
        turtle.width(width)
        turtle.goto(screen_p2[0], screen_p2[1])
        
        if self.animate:
            turtle.update()
            time.sleep(ANIMATION_DELAY)
        
        turtle.width(1)



    def run_episode(self, episode_num, show_visual=False):
        self.reset()
        total_reward = 0
        
        for iteration in range(self.max_iterations):
            # Get current state
            if len(self.nodes) == 0:
                continue
                
            current_node = self.nodes[-1] if len(self.nodes) > 1 else self.start
            nearest_to_goal = min(self.nodes, key=lambda n: self.dist(n, self.goal))
            
            # Create state representation
            state = self.agent.get_state_features(current_node, self.goal, nearest_to_goal)
            valid_actions = self.get_sampling_actions()
            
            # Choose action
            action = self.agent.choose_action(state, valid_actions)
            
            # Execute action to get a sampling point (always a cell center)
            random_point = self.sample_point_with_action(action)
            nearest_node = self.get_nearest_node(random_point)
            
            # Get direction from nearest node to random point
            direction = (random_point[0] - nearest_node[0], random_point[1] - nearest_node[1])
            distance = math.hypot(direction[0], direction[1])
            
            if distance == 0:
                reward = -0.1  # Small penalty for wasted iteration
                total_reward += reward
                continue
            
            # Calculate direction and normalize
            unit_direction = (direction[0] / distance, direction[1] / distance)
            
            # Instead of using step_size directly, find the closest cell center in the direction
            # This ensures we only move along cell centers
            
            # Try different step lengths to find a valid cell center
            valid_node_found = False
            
            # Try increasing distances to find a good cell center
            for step_mult in [1, 1.5, 2]:
                test_x = nearest_node[0] + step_mult * unit_direction[0]
                test_y = nearest_node[1] + step_mult * unit_direction[1]
                
                # Get the nearest cell center
                cell_x, cell_y = int(test_x), int(test_y)
                
                # Only consider cells that are free and not already visited
                if (0 <= cell_x < self.maze.cols and 0 <= cell_y < self.maze.rows and 
                    self.maze.is_free(cell_x, cell_y)):
                    
                    # Create new node at the exact cell center
                    new_node = (cell_x + 0.5, cell_y + 0.5)
                    
                    # Check if we can connect to this node without collision
                    if not self.check_collision(nearest_node, new_node) and new_node not in self.node_set:
                        valid_node_found = True
                        break
            
            # If no valid node found in the direct path, try neighboring cells
            if not valid_node_found:
                # Calculate base cell coordinates (where we wanted to go)
                base_x = int(nearest_node[0] + self.step_size * unit_direction[0])
                base_y = int(nearest_node[1] + self.step_size * unit_direction[1])
                
                # Try to find a nearby free cell, prioritizing those closer to the goal
                candidates = []
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        # Skip diagonal movements if either cardinal direction is blocked
                        if abs(dx) == 1 and abs(dy) == 1:
                            if not (self.maze.is_free(base_x + dx, base_y) and 
                                   self.maze.is_free(base_x, base_y + dy)):
                                continue
                                
                        nx, ny = base_x + dx, base_y + dy
                        if 0 <= nx < self.maze.cols and 0 <= ny < self.maze.rows:
                            if self.maze.is_free(nx, ny):
                                center = (nx + 0.5, ny + 0.5)
                                # Skip if already in tree or can't reach
                                if center in self.node_set or self.check_collision(nearest_node, center):
                                    continue
                                # Calculate priority based on distance to goal
                                dist_to_goal = self.dist(center, self.goal)
                                candidates.append((center, dist_to_goal))
                
                # Sort candidates by distance to goal
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_node = candidates[0][0]
                    valid_node_found = True
            
            # If still no valid node found, skip this iteration
            if not valid_node_found:
                reward = -0.5  # Larger penalty for hitting wall areas
                total_reward += reward
                continue
            
            # Calculate reward
            reward = self.calculate_reward(nearest_node, new_node, action, iteration)
            total_reward += reward
            
            # Check if node is valid
            if (new_node[0] < 0 or new_node[0] >= self.maze.cols or 
                new_node[1] < 0 or new_node[1] >= self.maze.rows or
                not self.maze.is_free(new_node[0], new_node[1]) or
                new_node in self.node_set or
                self.check_collision(nearest_node, new_node)):
                
                # Get next state for Q-learning update
                next_state = self.agent.get_state_features(current_node, self.goal, nearest_to_goal)
                self.agent.update_q_value(state, action, reward, next_state, valid_actions)
                continue
            
            # --- RRT* modifications start here ---
            
            # Find the best parent for the new node (lowest cost path)
            nearby_nodes = self.get_nearby_nodes(new_node, self.search_radius)
            best_parent = nearest_node
            best_cost = self.costs[nearest_node] + self.dist(nearest_node, new_node)
            
            # Check if there's a better path to this node
            for node in nearby_nodes:
                if not self.check_collision(node, new_node):
                    cost = self.costs[node] + self.dist(node, new_node)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = node
            
            # Add the node to the tree with the best parent
            self.nodes.append(new_node)
            self.node_set.add(new_node)
            self.parents[new_node] = best_parent
            self.costs[new_node] = best_cost
            
            if show_visual:
                self.draw_edge(best_parent, new_node)
            
            # Rewire the tree: check if the new node provides a better path for nearby nodes
            for node in nearby_nodes:
                # Skip if it's the parent of the new node
                if node == best_parent:
                    continue
                    
                # Check if routing through new_node creates a better path to node
                potential_cost = best_cost + self.dist(new_node, node)
                if potential_cost < self.costs[node] and not self.check_collision(new_node, node):
                    # Rewire: update parent and cost
                    old_parent = self.parents[node]
                    self.parents[node] = new_node
                    self.costs[node] = potential_cost
                    
                    if show_visual:
                        # If animating, remove old edge and draw new edge
                        self.draw_edge(old_parent, node, 'white', 1)  # "erase" old edge by drawing white
                        self.draw_edge(new_node, node, TREE_COLOR, 2)
            
            # --- RRT* modifications end here ---
            
            # Check if goal is reachable
            if (self.dist(new_node, self.goal) <= self.step_size and 
                not self.check_collision(new_node, self.goal)):
                
                # Success! Big reward
                reward += 100
                total_reward += reward
                
                # Reconstruct path
                path = []
                node = new_node
                while node is not None:
                    path.append(node)
                    node = self.parents.get(node)
                path.reverse()
                path.append(self.goal)
                
                if show_visual:
                    self.draw_edge(new_node, self.goal, color=PATH_COLOR, width=4)
                
                # Update Q-value for successful action
                next_state = self.agent.get_state_features(new_node, self.goal, new_node)
                self.agent.update_q_value(state, action, reward, next_state, [])
                
                return path, total_reward, iteration + 1
            
            # Update Q-value
            next_state = self.agent.get_state_features(new_node, self.goal, 
                                                     min(self.nodes, key=lambda n: self.dist(n, self.goal)))
            self.agent.update_q_value(state, action, reward, next_state, valid_actions)
        
        return None, total_reward, self.max_iterations

    def calculate_reward(self, old_node, new_node, action, iteration):
        """Calculate reward for the action taken."""
        reward = 0
        
        # Progress towards goal
        old_dist = self.dist(old_node, self.goal)
        new_dist = self.dist(new_node, self.goal)
        progress = old_dist - new_dist
        reward += progress * 10  # Reward getting closer to goal
        
        # Penalty for time
        reward -= 0.1  # Small penalty per iteration to encourage efficiency
        
        # Bonus for exploration (distance from other nodes)
        if len(self.nodes) > 1:
            min_dist_to_existing = min([self.dist(new_node, node) for node in self.nodes])
            if min_dist_to_existing > 2:
                reward += 1  # Exploration bonus
        
        # Action-specific rewards
        if action == 'goal_biased' and progress > 0:
            reward += 2  # Extra bonus for good goal-biased moves
        elif action == 'exploration' and len(self.nodes) < 50:
            reward += 0.5  # Exploration is good early on
        
        return reward

# === TRAINING AND VISUALIZATION ===
class RRTTrainer:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent = QLearningAgent()
        self.training_stats = {
            'episode_rewards': [],
            'path_lengths': [],
            'iterations_to_solve': [],
            'success_rate': []
        }
    
    def train(self, episodes=100, save_model=True):
        print(f"Training RRT* agent for {episodes} episodes...")
        
        successful_episodes = 0
        recent_success = []
        
        for episode in range(episodes):
            rrt = SmartRRTStar(self.maze, self.start, self.goal, self.agent, 
                          max_iterations=500, animate=False)
            
            path, reward, iterations = rrt.run_episode(episode, show_visual=False)
            
            self.training_stats['episode_rewards'].append(reward)
            
            if path:
                successful_episodes += 1
                recent_success.append(1)
                self.training_stats['path_lengths'].append(len(path))
                self.training_stats['iterations_to_solve'].append(iterations)
                print(f"Episode {episode+1}: SUCCESS! Path length: {len(path)}, Iterations: {iterations}, Reward: {reward:.2f}")
            else:
                recent_success.append(0)
                self.training_stats['path_lengths'].append(0)
                self.training_stats['iterations_to_solve'].append(iterations)
                print(f"Episode {episode+1}: Failed, Reward: {reward:.2f}")
            
            if len(recent_success) > 20:
                recent_success.pop(0)
            
            success_rate = sum(recent_success) / len(recent_success)
            self.training_stats['success_rate'].append(success_rate)
            
            self.agent.decay_epsilon()
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}: Success rate (last 20): {success_rate:.2%}, Epsilon: {self.agent.epsilon:.3f}")
        
        print(f"\nTraining completed!")
        print(f"Overall success rate: {successful_episodes/episodes:.2%}")
        
        if save_model:
            self.agent.save_model('rrt_star_q_model.pkl')
            print("Model saved to 'rrt_star_q_model.pkl'")
    
    def demonstrate(self, load_model=True):
        if load_model:
            if self.agent.load_model('rrt_star_q_model.pkl'):
                print("Loaded trained model")
            else:
                print("No saved model found, using current agent")
        
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        print("\nDemonstrating trained RRT* agent...")
        turtle.setup(COLS * CELL_SIZE + 100, ROWS * CELL_SIZE + 100)
        turtle.title("Smart RRT*")
        
        self.maze.draw(show_visual=True)
        
        rrt = SmartRRTStar(self.maze, self.start, self.goal, self.agent, 
                      max_iterations=1000, animate=True)
        
        path, reward, iterations = rrt.run_episode(0, show_visual=True)
        
        if path:
            print(f"Demonstration: SUCCESS! Path length: {len(path)}, Iterations: {iterations}")
            for i in range(len(path)-1):
                rrt.draw_edge(path[i], path[i+1], color=LEARNED_PATH_COLOR, width=6)
            
            path_length = sum(rrt.dist(path[i], path[i+1]) for i in range(len(path)-1))
            print(f"Optimal path length: {path_length:.2f}")
        else:
            print("Demonstration: Failed to find path")
        
        self.agent.epsilon = old_epsilon
        
        print("Click to close...")
        turtle.exitonclick()

# Add the original SmartRRT class back for comparison
class SmartRRT:
    def __init__(self, maze, start, goal, agent, step_size=1, max_iterations=1000, animate=True):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent = agent
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.animate = animate
        self.nodes = []
        self.parents = {}
        self.node_set = set()
        self.episode_rewards = []

    def reset(self):
        self.nodes = [self.start]
        self.parents = {self.start: None}
        self.node_set = {self.start}
        self.agent.action_history = []

    def dist(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_nearest_node(self, random_point):
        return min(self.nodes, key=lambda node: self.dist(node, random_point))
    
    def check_collision(self, p1, p2):
        """Stricter collision check that only allows direct connections 
        between cell centers with no diagonal movements through wall corners."""
        # Only cell centers should be connected (coordinates should be x.5, y.5)
        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(p2[0]), float(p2[1])
        
        # Get cell indices
        cell_x0, cell_y0 = int(x0), int(y0)
        cell_x1, cell_y1 = int(x1), int(y1)
        
        # If either endpoint is in a wall, return collision
        if not self.maze.is_free(cell_x0, cell_y0) or not self.maze.is_free(cell_x1, cell_y1):
            return True
            
        # Only allow direct connections between adjacent cells or the same cell
        dx = cell_x1 - cell_x0
        dy = cell_y1 - cell_y0
        
        # If it's the same cell, no collision
        if dx == 0 and dy == 0:
            return False
            
        # Only allow connections to adjacent cells (horizontally, vertically, or diagonally)
        if abs(dx) > 1 or abs(dy) > 1:
            return True
            
        # For diagonal movement, check if we can move through the corner
        if abs(dx) == 1 and abs(dy) == 1:
            # Check if both cardinal directions are free
            if not self.maze.is_free(cell_x1, cell_y0) or not self.maze.is_free(cell_x0, cell_y1):
                return True
                
        # Movement is valid
        return False
    
    def draw_edge(self, p1, p2, color=TREE_COLOR, width=2):
        if not self.animate:
            return
            
        screen_p1 = self.maze.grid_to_screen(p1[0], p1[1])
        screen_p2 = self.maze.grid_to_screen(p2[0], p2[1])
        
        turtle.up()
        turtle.goto(screen_p1[0], screen_p1[1])
        turtle.down()
        turtle.color(color)
        turtle.width(width)
        turtle.goto(screen_p2[0], screen_p2[1])
        
        if self.animate:
            turtle.update()
            time.sleep(ANIMATION_DELAY)
        
        turtle.width(1)
    
    def get_sampling_actions(self):
        return [
            'random',
            'goal_biased',
            'exploration',
            'path_optimized'
        ]
    
    def sample_point_with_action(self, action):
        def get_cell_center(x, y):
            return (x + 0.5, y + 0.5)
            
        if action == 'goal_biased':
            if random.random() < 0.3:  # Higher bias towards goal
                # Use exact goal coordinates
                return self.goal
            else:
                # Sample a free cell and return its center
                for _ in range(10):
                    col = random.randint(0, self.maze.cols-1)
                    row = random.randint(0, self.maze.rows-1)
                    if self.maze.is_free(col, row):
                        return get_cell_center(col, row)
                # Fallback if we can't find a free cell
                return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        elif action == 'exploration':
            # Sample points far from existing nodes, but at cell centers
            attempts = 0
            while attempts < 10:
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if not self.maze.is_free(col, row):
                    attempts += 1
                    continue
                    
                center = get_cell_center(col, row)
                min_dist = min([self.dist(center, node) for node in self.nodes])
                if min_dist > 3:  # Far from existing nodes
                    return center
                attempts += 1
                
            # Fallback to any free cell center
            for _ in range(5):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        elif action == 'path_optimized':
            # Sample near the current best path to goal, but at cell centers
            if len(self.nodes) > 5:
                # Pick a random existing node
                node = random.choice(self.nodes[-5:])  # Recent nodes
                
                # Find a free cell near the node
                best_center = None
                min_dist = float('inf')
                
                # Try several cells around the node
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        col = int(node[0]) + dx
                        row = int(node[1]) + dy
                        
                        if 0 <= col < self.maze.cols and 0 <= row < self.maze.rows:
                            if self.maze.is_free(col, row):
                                center = get_cell_center(col, row)
                                d = self.dist(center, self.goal)
                                if d < min_dist:
                                    min_dist = d
                                    best_center = center
                
                if best_center:
                    return best_center
            
            # Fallback to random free cell center
            for _ in range(5):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
        
        else:  # 'random'
            # Random, but still using cell centers
            for _ in range(10):
                col = random.randint(0, self.maze.cols-1)
                row = random.randint(0, self.maze.rows-1)
                if self.maze.is_free(col, row):
                    return get_cell_center(col, row)
            
            # Fallback
            return get_cell_center(random.randint(0, self.maze.cols-1), random.randint(0, self.maze.rows-1))
    
    def run_episode(self, episode_num, show_visual=False):
        self.reset()
        total_reward = 0
        
        for iteration in range(self.max_iterations):
            # Get current state
            if len(self.nodes) == 0:
                continue
                
            current_node = self.nodes[-1] if len(self.nodes) > 1 else self.start
            nearest_to_goal = min(self.nodes, key=lambda n: self.dist(n, self.goal))
            
            # Create state representation
            state = self.agent.get_state_features(current_node, self.goal, nearest_to_goal)
            valid_actions = self.get_sampling_actions()
            
            # Choose action
            action = self.agent.choose_action(state, valid_actions)
            
            # Execute action to get a sampling point (always a cell center)
            random_point = self.sample_point_with_action(action)
            nearest_node = self.get_nearest_node(random_point)
            
            # Get direction from nearest node to random point
            direction = (random_point[0] - nearest_node[0], random_point[1] - nearest_node[1])
            distance = math.hypot(direction[0], direction[1])
            
            if distance == 0:
                reward = -0.1  # Small penalty for wasted iteration
                total_reward += reward
                continue
            
            # Calculate direction and normalize
            unit_direction = (direction[0] / distance, direction[1] / distance)
            
            # Instead of using step_size directly, find the closest cell center in the direction
            # This ensures we only move along cell centers
            
            # Try different step lengths to find a valid cell center
            valid_node_found = False
            
            # Try increasing distances to find a good cell center
            for step_mult in [1, 1.5, 2]:
                test_x = nearest_node[0] + step_mult * unit_direction[0]
                test_y = nearest_node[1] + step_mult * unit_direction[1]
                
                # Get the nearest cell center
                cell_x, cell_y = int(test_x), int(test_y)
                
                # Only consider cells that are free and not already visited
                if (0 <= cell_x < self.maze.cols and 0 <= cell_y < self.maze.rows and 
                    self.maze.is_free(cell_x, cell_y)):
                    
                    # Create new node at the exact cell center
                    new_node = (cell_x + 0.5, cell_y + 0.5)
                    
                    # Check if we can connect to this node without collision
                    if not self.check_collision(nearest_node, new_node) and new_node not in self.node_set:
                        valid_node_found = True
                        break
            
            # If no valid node found in the direct path, try neighboring cells
            if not valid_node_found:
                # Calculate base cell coordinates (where we wanted to go)
                base_x = int(nearest_node[0] + self.step_size * unit_direction[0])
                base_y = int(nearest_node[1] + self.step_size * unit_direction[1])
                
                # Try to find a nearby free cell, prioritizing those closer to the goal
                candidates = []
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        # Skip diagonal movements if either cardinal direction is blocked
                        if abs(dx) == 1 and abs(dy) == 1:
                            if not (self.maze.is_free(base_x + dx, base_y) and 
                                   self.maze.is_free(base_x, base_y + dy)):
                                continue
                                
                        nx, ny = base_x + dx, base_y + dy
                        if 0 <= nx < self.maze.cols and 0 <= ny < self.maze.rows:
                            if self.maze.is_free(nx, ny):
                                center = (nx + 0.5, ny + 0.5)
                                # Skip if already in tree or can't reach
                                if center in self.node_set or self.check_collision(nearest_node, center):
                                    continue
                                # Calculate priority based on distance to goal
                                dist_to_goal = self.dist(center, self.goal)
                                candidates.append((center, dist_to_goal))
                
                # Sort candidates by distance to goal
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    new_node = candidates[0][0]
                    valid_node_found = True
            
            # If still no valid node found, skip this iteration
            if not valid_node_found:
                reward = -0.5  # Larger penalty for hitting wall areas
                total_reward += reward
                continue
            
            # Calculate reward
            reward = self.calculate_reward(nearest_node, new_node, action, iteration)
            total_reward += reward
            
            # Check if node is valid
            if (new_node[0] < 0 or new_node[0] >= self.maze.cols or 
                new_node[1] < 0 or new_node[1] >= self.maze.rows or
                not self.maze.is_free(new_node[0], new_node[1]) or
                new_node in self.node_set or
                self.check_collision(nearest_node, new_node)):
                
                # Get next state for Q-learning update
                next_state = self.agent.get_state_features(current_node, self.goal, nearest_to_goal)
                self.agent.update_q_value(state, action, reward, next_state, valid_actions)
                continue
            
            # Add successful node
            self.nodes.append(new_node)
            self.node_set.add(new_node)
            self.parents[new_node] = nearest_node
            
            if show_visual:
                self.draw_edge(nearest_node, new_node)
            
            # Check if goal is reachable
            if (self.dist(new_node, self.goal) <= self.step_size and 
                not self.check_collision(new_node, self.goal)):
                
                # Success! Big reward
                reward += 100
                total_reward += reward
                
                # Reconstruct path
                path = []
                node = new_node
                while node is not None:
                    path.append(node)
                    node = self.parents.get(node)
                path.reverse()
                path.append(self.goal)
                
                if show_visual:
                    self.draw_edge(new_node, self.goal, color=PATH_COLOR, width=4)
                
                # Update Q-value for successful action
                next_state = self.agent.get_state_features(new_node, self.goal, new_node)
                self.agent.update_q_value(state, action, reward, next_state, [])
                
                return path, total_reward, iteration + 1
            
            # Update Q-value
            next_state = self.agent.get_state_features(new_node, self.goal, 
                                                     min(self.nodes, key=lambda n: self.dist(n, self.goal)))
            self.agent.update_q_value(state, action, reward, next_state, valid_actions)
        
        return None, total_reward, self.max_iterations
    
    def calculate_reward(self, old_node, new_node, action, iteration):
        """Calculate reward for the action taken."""
        reward = 0
        
        # Progress towards goal
        old_dist = self.dist(old_node, self.goal)
        new_dist = self.dist(new_node, self.goal)
        progress = old_dist - new_dist
        reward += progress * 10  # Reward getting closer to goal
        
        # Penalty for time
        reward -= 0.1  # Small penalty per iteration to encourage efficiency
        
        # Bonus for exploration (distance from other nodes)
        if len(self.nodes) > 1:
            min_dist_to_existing = min([self.dist(new_node, node) for node in self.nodes])
            if min_dist_to_existing > 2:
                reward += 1  # Exploration bonus
        
        # Action-specific rewards
        if action == 'goal_biased' and progress > 0:
            reward += 2  # Extra bonus for good goal-biased moves
        elif action == 'exploration' and len(self.nodes) < 50:
            reward += 0.5  # Exploration is good early on
        
        return reward

# Original RRT Trainer class for comparison
class OriginalRRTTrainer:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent = QLearningAgent()
        
    def demonstrate(self):
        print("\nDemonstrating original RRT algorithm...")
        turtle.setup(COLS * CELL_SIZE + 100, ROWS * CELL_SIZE + 100)
        turtle.title("Original RRT")
        
        self.maze.draw(show_visual=True)
        
        rrt = SmartRRT(self.maze, self.start, self.goal, self.agent, 
                      max_iterations=1000, animate=True)
        
        path, reward, iterations = rrt.run_episode(0, show_visual=True)
        
        if path:
            print(f"Original RRT: SUCCESS! Path length: {len(path)}, Iterations: {iterations}")
            for i in range(len(path)-1):
                rrt.draw_edge(path[i], path[i+1], color='purple', width=4)
            
            path_length = sum(rrt.dist(path[i], path[i+1]) for i in range(len(path)-1))
            print(f"Path length: {path_length:.2f}")
        else:
            print("Original RRT: Failed to find path")
        
        print("Click to close...")
        turtle.exitonclick()

# === MAIN ===
if __name__ == '__main__':
    start_x, start_y = idx_to_xy(START_IDX)
    goal_x, goal_y = idx_to_xy(END_IDX)
    
    start_xy = (start_x + 0.5, start_y + 0.5)
    goal_xy = (goal_x + 0.5, goal_y + 0.5)
    
    print(f"Start coordinates: {start_xy}")
    print(f"Goal coordinates: {goal_xy}")
    
    maze = Maze(TILES, COLS, ROWS, CELL_SIZE, START_IDX, END_IDX)
    
    print("\nChoose an option:")
    print("1. Train new RRT* model (100 episodes)")
    print("2. Load existing RRT* model and demonstrate")
    print("3. Train new RRT* model and demonstrate")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        trainer = RRTTrainer(maze, start_xy, goal_xy)
        trainer.train(episodes=100)
    elif choice == '2':
        trainer = RRTTrainer(maze, start_xy, goal_xy)
        trainer.demonstrate(load_model=True)
    elif choice == '3':
        trainer = RRTTrainer(maze, start_xy, goal_xy)
        trainer.train(episodes=100)
        trainer.demonstrate(load_model=False)
    else:
        print("Invalid choice, running demonstration with existing model...")
        trainer = RRTTrainer(maze, start_xy, goal_xy)
        trainer.demonstrate(load_model=True)