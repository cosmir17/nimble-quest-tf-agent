from enum import Enum


class GameStage(Enum):
    in_progress = 0
    game_over = 1
    starting_page = 2
    interval = 3

    character_upgrade = 4
    interval_sorry = 5
    interval_upgrade = 6
    paused_game_while_in_progress = 7
    game_over_sorry = 8
    store_page = 9
    main_page = 10

    died = 11
