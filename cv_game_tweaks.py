from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.dealer import Dealer
from pypokerengine.engine.table import Table
from pypokerengine.engine.deck import Deck


class EmptyCheatDeck(Deck):

    def __init__(self):
        super()
        self.deck = []

    def add_card(self, card_id):
        self.deck.append(card_id)


class CheatDealer(Dealer):

    def __init__(self, small_blind_amount=None, initial_stack=None, ante=None):
        super().__init__(self, small_blind_amount=small_blind_amount, initial_stack=initial_stack, ante=ante)
        self.table = Table(cheat_deck=EmptyCheatDeck())


def start_cv_poker(config, verbose=2):
    config.validation()
    dealer = CheatDealer(config.sb_amount, config.initial_stack, config.ante)
    number_of_players = len(config.players_info)
    for player_num in range(2 * number_of_players):
        dealer.table.deck.add_card('XX')
    dealer.set_verbose(verbose)
    dealer.set_blind_structure(config.blind_structure)
    for info in config.players_info:
        dealer.register_player(info["name"], info["algorithm"])
    result_message = dealer.start_game(config.max_round)
    return _format_result(result_message)


def _format_result(result_message):
    return {
            "rule": result_message["message"]["game_information"]["rule"],
            "players": result_message["message"]["game_information"]["seats"]
            }


class CVRoundManager(RoundManager):

    @classmethod
    def start_new_round(self, round_count, small_blind_amount, ante_amount, table):
        _state = self.__gen_initial_state(round_count, small_blind_amount, table)
        state = self.__deep_copy_state(_state)
        table = state["table"]

        self.__correct_ante(ante_amount, table.seats.players)
        self.__correct_blind(small_blind_amount, table)
        self.__deal_holecard(table.deck, table.seats.players)
        start_msg = self.__round_start_message(round_count, table)
        state, street_msgs = self.__start_street(state)
        return state, start_msg + street_msgs

