import random
import numpy as np

class Bar:
	def __init__(self, bar_min=0, bar_max=100, target=50):
		print('Bar init...',end='')
		self._nb_point = 40
		self._position = 0
		self._bar_min = bar_min
		self._bar_max = bar_max
		self._target = target
		self._moves = {0:'nothing', 1:'left', 2:'right'}
		print('[target:{}]'.format(self._target),end='')
		print('ok')
		self.display_bar(self._position)


	def display_bar(self, pos):
		pos = int(pos*(self._nb_point/(self._bar_max-self._bar_min)))
		line = '.'*self._nb_point
		line = list(line)
		if pos < 0:
			line[0] = 'o'
		elif pos >= len(line):
			line[len(line)-1] = 'o'
		else:
			line[pos] = 'o'
		
		line = ''.join(line)
		line = str(self._bar_min)+'|'+line+'|'+str(self._bar_max)
		print(line)


	def make_move(self, move):
		prev_state = self._position

		if move in self._moves:
			if move == 1:
				self._position -= 10
			elif move == 2:
				self._position += 10

		self._position = min(self._position, self._bar_max)
		self._position = max(self._position, self._bar_min)
		self.display_bar(self._position)
		
		prev_delta = abs(prev_state - self._target)
		new_delta = abs(self._position - self._target)
		
		if new_delta < prev_delta:
			reward = 1
		else:
			reward = -1
		
		#reward = (self._bar_max - self._bar_min)/2 - delta
		state = {
			'reward':reward,
			'prev_state':prev_state,
			'new_state':self._position,
			'action':move
		}
		return state

	@property
	def current_state(self):
		return self._position

	@property
	def moves(self):
		return self._moves

	@property
	def state_space_range(self):
		return (self._bar_min,self._bar_max)
	

class Droid:
	def __init__(self, possible_moves, state_space_range):
		print('Droid init...',end='')
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense
		self._possible_moves = possible_moves
		self.state_space_range = state_space_range

		self._brain = Sequential()
		self._brain.add(Dense(10, input_dim=1, activation='sigmoid'))
		self._brain.add(Dense(10, activation='sigmoid'))
		self._brain.add(Dense(3, activation='sigmoid'))
		self._brain.compile(loss='mse', optimizer='adam')

		self.memory = []
		self.save_before_training = 10
		self.saving_done = 0
		self.lr = 0.01
		self.discount = 1.0
		self.min_discount = 0.1
		self.discount_factor = 0.95

		print('ok')

	def save_state(self, state_data):
		self.memory.append(state_data)
		self.saving_done += 1
		if self.saving_done % self.save_before_training == 0:
			self._train()

	def what_do_you_think(self, current_state):
		if random.random() < self.discount:
			print('[exploration {}]'.format(self.discount),end='')
			return list(self._possible_moves.keys())[random.randint(0,len(self._possible_moves)-1)]
		else:
			print('[knowledge]{}'.format(self.discount),end='')
			return np.argmax(self._brain.predict([current_state]))

	def get_scaled_state_space(self, data):
		return data/(self.state_space_range[1]-self.state_space_range[0])

	def _train(self):
		print('Droid: training...',end='')

		for state in self.memory:
			#state['prev_state'] = self.get_scaled_state_space(state['prev_state'])
			#state['new_state'] = self.get_scaled_state_space(state['new_state'])
			#idx = np.argmax(self._brain.predict([state['new_state']])[0])
			max_future_reward = np.max(self._brain.predict([state['new_state']])[0])
			current_predicted_value = self._brain.predict([state['prev_state']])[0]
			current_predicted_value_for_value = current_predicted_value[state['action']]
			delta = self.lr*(max_future_reward - current_predicted_value_for_value)
			
			#new_value = current_predicted_value[state['action']]*(1-self.lr) + self.lr*(state['reward']+max_future_reward)
			# print("pred_old {}, pred_new {}".format(pred[idx], new_value))
			current_predicted_value[state['action']] += delta
			print('state_action {}, current_pred {}'.format(state['action'], current_predicted_value))
			self._brain.fit(np.array([state['prev_state']]), np.array([current_predicted_value]), verbose=0, epochs=1)

			#lr*((R+discount*max(max_future_reward)) - current_predicted) # DQN Formula

		self.discount *= self.discount_factor
		if self.discount < self.min_discount:
			self.discount = self.min_discount
		
		print('ok')
		self.memory.clear()

def plot_data(data):
	import matplotlib.pyplot as plt
	new_states = [state['new_state'] for state in data]
	x_axis = list(range(len(new_states)))
	rewards = [state['reward'] for state in data]
	#plt.plot([state['it'] for state in data])

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('discount', color=color)
	ax1.plot([state['discount'] for state in data])
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	ax2.set_ylabel('states', color=color)  # we already handled the x-label with ax1
	ax2.plot(new_states)
	ax2.scatter(x_axis, new_states, c=rewards)
	ax2.tick_params(axis='y', labelcolor=color)

	plt.legend(['discount','states'])
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()
	

if __name__ == '__main__':
	#bar = Bar(target=random.randint(0,100))
	target = 50
	bar = Bar(target=target)
	droid = Droid(bar.moves, bar.state_space_range)
	history = []
	it = 1000
	for i in range(it):
		action_to_do = droid.what_do_you_think(bar.current_state)
		state = bar.make_move(action_to_do)
		state['discount'] = droid.discount
		state['it'] = i
		history.append(state)
		print('r:',state)
		droid.save_state(state)
	print("Iteration: {}, Target: {}, Last position: {}, LR: {}, Discount: {}".format(it, target, bar._position, droid.lr, droid.discount))
	plot_data(history)
	