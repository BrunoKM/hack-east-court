from pypokerengine.api.game import setup_config
from player import ConsolePlayer
from cv_game_tweaks import start_cv_poker


def configure_game(number_of_players=2, max_round=10, initial_stack=100, small_blind_amount=5):
    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
    for player in range(number_of_players):
        config.register_player(name="p" + str(player), algorithm=ConsolePlayer())
    print(start_cv_poker(config, verbose=1))


configure_game()


