from chess import Board, Move, BB_D5, BB_D8, D8

from model import ChessModel

# m = ChessModel("savedmodel.pt")
#
# encoded_moves = m.generate([])




b = Board()

print(b.push_san("e4"))
print(b.push_san("d5"))

for m
# print(b.push_san("e4d5"))

print(b)

for m in b.generate_legal_moves(to_mask=BB_D5):
    print(m)
    print(b.piece_at(m.from_square))
    print(b.piece_at(D8).piece_type)

# for m in b.generate_legal_moves():
#     b.
#     print(m)

