import re

move_regex = re.compile(r"(?P<movenum>[0-9]+)\. "
                        r"((([BNRKQ][1-8]?)?[abcdefgh]?[1-8]?x?[abcdefgh][1-8]|O-O-O|O-O)(=[BNRKQ])?[+#]?[?!]{0,2}) "
                        r"((([BNRKQ][1-8]?)?[abcdefgh]?[1-8]?x?[abcdefgh][1-8]|O-O-O|O-O)(=[BNRKQ])?[+#]?[?!]{0,2})?")


def parse_regex(game):
    # tokens = []
    move_num = 0
    for match in re.finditer(move_regex, game):
        this_move_num = int(match.group("movenum"))
        # if this_move_num != move_num + 1:

        #     print(f"Couldn't find move {move_num + 1}")
        #     print("Game: " + game)
        #     break
        move_num = this_move_num
    # print(tokens)
    # return tokens
    return move_num == game.count(".")
