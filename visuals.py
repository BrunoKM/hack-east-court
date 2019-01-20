import arcade
import random

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 780
COIN_COUNT = 100
MOVEMENT_SPEED = 5
CARD1=0
CARD2=1


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height):
        super().__init__(width, height)

        # Sprite lists
        self.player1_list = None
        self.player2_list = None
        self.game_list = None
        self.p1_card_list=None
        self.p2_card_list=None

        # Set up the player
        self.score = 0
        self.player_sprite = None
        self.physics_engine = None

        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        # Create the sprite lists
        self.player1_list = arcade.SpriteList()
        self.player2_list = arcade.SpriteList()
        self.game_list = arcade.SpriteList()
        self.p1_card_list = arcade.SpriteList()
        self.p2_card_list = arcade.SpriteList()

        # Score
        self.score = 0

        # Initialize all cards for each player
        suits = ['s', 'd', 'c', 'h']
        values = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
        for suit in suits:
            for value in values:
                card = arcade.Sprite("card-BMPS/%s%s.bmp" % (suit, value))
                card.center_x = 500
                card.center_y = 1000
                self.p1_card_list.append(card)
                self.p2_card_list.append(card)

        # Set up the player1 display



        # p1_first_likely_card1 = arcade.Sprite("card-BMPS/c01.bmp")
        # p1_first_likely_card1.center_x = 50
        # p1_first_likely_card1.center_y = 700
        # self.player1_list.append(p1_first_likely_card1)
        # p1_first_likely_card2 = arcade.Sprite("card-BMPS/d01.bmp")
        # p1_first_likely_card2.center_x = 150
        # p1_first_likely_card2.center_y = 700
        # self.player1_list.append(p1_first_likely_card2)


        # Set up player 2 display
        # p2_first_likely_card1 = arcade.Sprite("card-BMPS/h01.bmp")
        # p2_first_likely_card1.center_x = 1150
        # p2_first_likely_card1.center_y = 700
        # self.player1_list.append(p2_first_likely_card1)
        # p2_first_likely_card2 = arcade.Sprite("card-BMPS/s01.bmp")
        # p2_first_likely_card2.center_x = 1050
        # p2_first_likely_card2.center_y = 700
        # self.player1_list.append(p2_first_likely_card2)






        pass


        # # Character image from kenney.nl
        # self.player_sprite = arcade.Sprite("henners.png", 0.2)
        # self.player_sprite.center_x = 50  # Starting position
        # self.player_sprite.center_y = 50
        # self.player_list.append(self.player_sprite)
        #
        # # Create the coins
        # for i in range(COIN_COUNT):
        #     # Create the coin instance
        #     # Coin image from kenney.nl
        #     coin = arcade.Sprite("coin.png", 0.05)
        #
        #     # Position the coin
        #     coin.center_x = random.randrange(SCREEN_WIDTH)
        #     coin.center_y = random.randrange(SCREEN_HEIGHT)
        #
        #     # Add the coin to the lists
        #     self.coin_list.append(coin)
        # pass


    def on_draw(self):
        """ Render the screen. """
        arcade.start_render()
        self.player1_list.draw()
        self.player2_list.draw()
        self.game_list.draw()
        self.p1_card_list.draw()
        self.p2_card_list.draw()
        # draw text etc
        arcade.draw_text("Player 1", start_x=20, start_y=760, color=arcade.color.AFRICAN_VIOLET, font_size=15, bold=True)
        arcade.draw_text("Player 2", start_x=1100, start_y=760, color=arcade.color.AFRICAN_VIOLET, font_size=15,
                         bold=True)

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        self.p1_card_list.move(500, 1000)
        self.p2_card_list.move(500, 1000)


        f=open("inputcards.txt","r")
        card_string=f.readline()
        input_list = list(map(int, card_string.split(',')))

        p1_card1 = self.p1_card_list[input_list[0]]
        p1_card1.set_position(50,700)

        p1_card2 = self.p1_card_list[input_list[1]]
        p1_card2.set_position(150,700)

        p2_card1 = self.p2_card_list[input_list[2]]
        p2_card1.set_position(1050, 700)

        p2_card2 = self.p2_card_list[input_list[3]]
        p2_card2.set_position(1150, 700)



    # def on_key_release(self, symbol, modifiers):
    #     """Called on key release"""
    #     self.p1_card_list.move(500,1000)\



    def update(self, delta_time):
        """ All the logic to move, and the game logic goes here. """

        self.p1_card_list.update()
        self.p2_card_list.update()




def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()