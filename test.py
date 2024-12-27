from io import StringIO

import chess.pgn
from chess import Board

from model import ChessModel, Medium
from play_chars import find_next_move
from tokenisation.charactertokeniser import encode, decode

m = ChessModel("stockfishmodel_chars.pt", vocab_size=44, params=Medium, device="cpu")

game = ("1.e4 e6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Be7 5.Nf3 Nf6 6.Nxf6+ Bxf6 7.Bd3 Nd7 8.Qe2 c5 9.Be3 Qa5+ 10.c3 cxd4 "
        "11.Nxd4 O-O 12.Rd1 a6 13.Nf3 Qxa2 14.h4 Qa5 15.Ng5 h6 16.Bh7+ Kh8 17.Bb1 Qb5 18.Qh5 Kg8 19.Bh7+ Kh8 20.Bd3 "
        "Qxb2 21.Nxf7+ Rxf7 22.Qxf7 Bxc3+ 23.Kf1 Qa3 24.Bxh6 Qf8 25.Qg6 Nf6 26.Bg5 Kg8 27.Bxf6 Bxf6 28.Qh7+ Kf7 "
        "29.Qh5+ Ke7 30.Qc5+ Ke8 31.Bg6+ Qf7 32.Qh5 Qxg6 33.Qxg6+ Ke7 34.Kg1 Be5 35.Qg5+ Bf6 36.Qa5 b6 37.Qa3+ Ke8 "
        "38.Qb3 Bd7 39.h5 Rc8 40.h6 g5 41.h7 Bh8 42.Rh6 b5 43.Rxe6+ Kd8 44.Qd5 Kc7 45.Qxd7+ Kb8 46.Rb6+ Ka8 47.")


board = Board(fen="k1r4b/3Q3P/pR6/1p4p1/8/8/5PP1/3R2K1 w - - 3 47")

start_position = encode(game)

print("Starting position (", len(start_position), ")", start_position)

for x in range(1, 100):
    tokens = m.generate(start_position, num_moves_to_generate=5)

    next_move = find_next_move(board, tokens, len(start_position))

    print("Generated tokens  (", len(tokens), ")", tokens)
    print("Decoded tokens (", len(decode(tokens)), ")", decode(tokens))
    print("Next move is", next_move)
    # if next_move is not None and next_move[1] is not None:
    #     break
