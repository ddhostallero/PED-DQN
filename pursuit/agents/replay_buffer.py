from collections import deque
import random

class ReplayBuffer:
	def __init__(self, capacity):
		self.replay_memory = deque(maxlen=capacity)

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self, batch_size):
		return random.sample(self.replay_memory, batch_size)
