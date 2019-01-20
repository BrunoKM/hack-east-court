from pypokerengine.engine.deck import Deck
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
import matplotlib.pyplot as plt


def determine_hand_strength_dist(community_card_list):
    prob_dict = {}
    deck = Deck()
    for card in community_card_list:
        deck.deck.remove(card)

    for possible_card_a in deck.deck:
        for possible_card_b in deck.deck:
            hand_strength = HandEvaluator.gen_hand_rank_info([possible_card_a, possible_card_b], community_card_list)['hand']['strength']
            if hand_strength == 'FLASH':
                hand_strength = 'FLUSH'
            if hand_strength == 'STRAIGHTFLASH':
                hand_strength = 'STRAIGHT FLUSH'

            if hand_strength in prob_dict:
                prob_dict[hand_strength] += 1
            else:
                prob_dict[hand_strength] = 1
    for key, value in prob_dict.items():
        prob_dict[key] = value / (len(deck.deck) ** 2)

    sorted_probs = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)

    value_list = [tuple[0] for tuple in sorted_probs]
    prob_list = [tuple[1] for tuple in sorted_probs]

    return value_list, prob_list


def get_best_hand(community_card_list):
    hand_strength_set = {0}
    deck = Deck()
    for card in community_card_list:
        deck.deck.remove(card)
    for possible_card_a in deck.deck:
        deck.deck.remove(possible_card_a)
        for possible_card_b in deck.deck:
            hand_strength = HandEvaluator.eval_hand([possible_card_a, possible_card_b], community_card_list)
            if hand_strength > max(hand_strength_set):
                nuts_cards = {(str(possible_card_a), str(possible_card_b))}
            elif hand_strength == max(hand_strength_set):
                if not (str(possible_card_b), str(possible_card_a)) in nuts_cards:
                    nuts_cards.add((str(possible_card_a), str(possible_card_b)))
            hand_strength_set.add(hand_strength)
        deck.deck = [possible_card_a] + deck.deck
    return nuts_cards

# 'Random' cards
com_cards = [Card(8, 11), Card(2, 9), Card(4, 8), Card(2, 12), Card(2, 5)]

# Straight flush
# com_cards = [Card(4, 12), Card(4, 13), Card(4, 14), Card(4, 11), Card(4, 10)]

# 3 Aces
# com_cards = [Card(2, 14), Card(4, 14), Card(8, 14)]

print(get_best_hand(com_cards))

val, prob = determine_hand_strength_dist(com_cards)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(val, prob, 1/len(val), color='r')
plt.show()
