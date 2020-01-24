import random
from DeepRTS import Engine
from DeepRTS.python import scenario
from DeepRTS.python import Config
from DeepRTS.python import Game

episodes = 100
steps = 100

MAP="32.json" 

gui_config = Config(
            render=True,
            view=True,
            inputs=False,
            caption=False,
            unit_health=True,
            unit_outline=True,
            unit_animation=True,
            audio=False,
            audio_volume=50
        )


engine_config = Engine.Config.defaults()

game = Game(
            MAP,
            n_players = 1,
            engine_config = engine_config,
            gui_config = gui_config,
            terminal_signal = False
        )
game.set_max_fps(1000000)
game.set_max_ups(1000000)


for ep in range(episodes):
    print("Episode " + str(ep + 1))
    game.start()
    game.reset()

    for step in range(steps):
        print("Step " + str(step + 1))
        player = game.players[0]
        action = random.randint(0, 15)
        player.do_action(action + 1)

        game.update()
        state = game.get_state()

        game.render()
        game.view()

        print("Current state: ")
        print(state)
        print("FPS: " + str(game.get_fps()))

