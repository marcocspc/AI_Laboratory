import numpy
import time
from pyDeepRTS import Game as PyDeepRTS

'''
    pyDeepRTS.Game properties:

    ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_caption', '_on_episode_end', '_on_episode_start', '_on_tile_deplete', '_on_unit_create', '_on_unit_destroy', '_render', '_update', 'add_player', 'caption', 'config', 'get_episode', 'get_episode_duration', 'get_fps', 'get_game', 'get_height', 'get_id', 'get_max_fps', 'get_max_ups', 'get_ticks', 'get_ticks_modifier', 'get_ups', 'get_width', 'is_terminal', 'map', 'players', 'render', 'reset', 'set_max_fps', 'set_max_ups', 'start', 'state', 'stop', 'tick', 'tilemap', 'units', 'update']

'''

# Start the game
g = PyDeepRTS('10x10-2v2.json')

print("Running DeepRTS on 10x10-2v2.json map.")

# Add players
player1 = g.add_player()
player2 = g.add_player()

# Set FPS and UPS limits
g.set_max_fps(10000000)
g.set_max_ups(10000000)

# Start the game (flag)
g.start()

# Run forever
while True:
    g.tick()  # Update the game clock
    g.update()  # Process the game state
    g.render()  # Draw the game state to graphics
    state = g.state  # Captures current state (Returns None if .capture_every is set for some iterations)
    g.caption()  # Show Window caption
    
    # If the game is in terminal state
    if g.is_terminal():
        g.reset()  # Reset the game

    # Perform random action for player 1
    player1.do_action(numpy.random.randint(0, 16))
    
    # Perform random action for player 2
    player2.do_action(numpy.random.randint(0, 16))
