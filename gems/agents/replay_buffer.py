from collections import deque
import random

class ReplayBuffer:
	def __init__(self, capacity):
		self.replay_memory = deque(maxlen=capacity)

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self, batch_size):
		return random.sample(self.replay_memory, batch_size)

class Rollout:
	def __init__(self, fields):
		self.fields = fields
		self.reset()

	def add_to_memory(self, experience):
		for key, value in experience.iteritems():
			self.memory[key].append(value)

	def get_rollout(self):
		return self.memory

	def reset(self):
		self.memory = {}
		for f in self.fields:
			self.memory[f] = []