from game.game import Move, Game, Winner, Ply


def parse(game: str) -> Game:
    moves = []

    # Accumulator for the current number
    cur_num = 0

    move_num = 0
    piece = 'x'
    from_file = 'x'
    from_rank = 'x'
    file = 'x'
    rank = 'x'
    take = False
    check = False
    short_castle = False
    long_castle = False
    analysis = 0

    white = False
    black = False

    castle_os = 0
    promotion = False
    promotion_to = 'x'
    check_mate = False

    # Game flags
    game_over = False
    winner = Winner.NONE

    white_ply = None

    i = 0
    while i < len(game):
        cur_char = game[i]
        if cur_char in "0123456789":
            cur_num = cur_num * 10 + int(cur_char)
        elif cur_char == '.':
            move_num = cur_num
        if white:
            if cur_char in "BNQRK":
                if promotion:
                    promotion_to = cur_char
                else:
                    piece = cur_char
            elif cur_char in "abcdefgh":
                from_file = file
                file = cur_char
                if piece == 'x':
                    piece = 'P'
            elif cur_char in "12345678":
                from_rank = rank
                rank = cur_char
            elif cur_char == 'x':
                take = True
            elif cur_char == '+':
                check = True
            elif cur_char == 'O':
                castle_os += 1
            elif cur_char == '=':
                promotion = True
            elif cur_char == '?':
                if i + 1 < len(game) and game[i + 1] == '!':
                    analysis = -1
                    i += 1
                elif i + 1 < len(game) and game[i + 1] == '?':
                    analysis = -4
                    i += 1
                else:
                    analysis = -2
            elif cur_char == '!':
                if i + 1 < len(game) and game[i + 1] == '?':
                    analysis = 0
                    i += 1
                elif i + 1 < len(game) and game[i + 1] == '!':
                    analysis = 4
                    i += 1
                else:
                    analysis = 2
        if cur_char == '#':
            check_mate = True
        elif cur_char == '/':
            winner = Winner.DRAW
            break
        elif cur_char == '-':
            if castle_os == 0:
                winner = Winner.WHITE if cur_num == 1 else Winner.BLACK
                break
        elif cur_char == ' ':
            if castle_os == 3:
                long_castle = True
            elif castle_os == 2:
                short_castle = True
            if white:
                new_ply = Ply(piece, from_file, from_rank, file, rank, take, check, short_castle, long_castle, promotion_to, check_mate, analysis)
                piece = 'x'
                from_file = 'x'
                from_rank = 'x'
                file = 'x'
                rank = 'x'
                take = False
                check = False
                short_castle = False
                long_castle = False
                promotion_to = 'x'
                analysis = 0
                if black:
                    moves.append(Move(move_num, white_ply, new_ply))
                    white_ply = None
                    white = False
                    black = False
                    move_num = 0
                else:
                    white_ply = new_ply
                    black = True
            elif move_num > 0:
                white = True
            cur_num = 0
            castle_os = 0
            promotion = False
        i += 1

    if white_ply is not None:
        moves.append(Move(move_num, white_ply))

    # for move in moves:
    #     if move.white is not None:
    #         all_moves.append(move.white)
    #     if move.black is not None:
    #         all_moves.append(move.black)

    return Game(moves, winner)
    # games_match = str(new_game) == game
    # if not games_match:
    #     print(game)
    #     print(new_game)
    # return True
