import chess
from chess import Move, Piece, Board

MAX_MOVES: int = 149
# Space represents 0 which shouldn't ever be used - we just use printable ascii chars to tokenise
FIRST_NUM: int = ord(' ')
LAST_NUM: int = ord(' ') + MAX_MOVES
SHORT_CASTLE: int = LAST_NUM + 1
LONG_CASTLE: int = SHORT_CASTLE + 1
PAWN: int = LONG_CASTLE + 1
BISHOP: int = LONG_CASTLE + 2
KNIGHT: int = LONG_CASTLE + 3
QUEEN: int = LONG_CASTLE + 4
ROOK: int = LONG_CASTLE + 5
KING: int = LONG_CASTLE + 6
TAKE: int = KING + 1
CHECK: int = TAKE + 1
FIRST_POSITION: int = CHECK + 1
LAST_POSITION: int = FIRST_POSITION + 64


def decode_position(b):
    val = b - FIRST_POSITION
    file = chr(ord('a') + (val % 8))
    rank = int(val / 8) + 1
    return f"{file}{rank}"


def decode_token(b: int) -> str:
    if FIRST_NUM <= b <= LAST_NUM:
        return f"{b - FIRST_NUM}."
    elif b == SHORT_CASTLE:
        return f"O-O"
    elif b == LONG_CASTLE:
        return f"O-O-O"
    elif b == PAWN:
        return f"P"
    elif b == BISHOP:
        return f"B"
    elif b == KNIGHT:
        return f"N"
    elif b == QUEEN:
        return f"Q"
    elif b == ROOK:
        return f"R"
    elif b == KING:
        return f"K"
    elif b == TAKE:
        return f"x"
    elif b == CHECK:
        return f"+"
    elif b >= FIRST_POSITION:
        return decode_position(b)
    else:
        return f"{b}"


def decode(token_data: []) -> str:

    output = []
    for b in token_data:
        output.append(decode_token(b))

    return " ".join(output)


encode_piece_map = {
    chess.PAWN: PAWN,
    chess.BISHOP: BISHOP,
    chess.KNIGHT: KNIGHT,
    chess.QUEEN: QUEEN,
    chess.ROOK: ROOK,
    chess.KING: KING
}


def encode(board: Board, move: Move) -> [int]:
    tokens = []
    if board.is_kingside_castling(move):
        tokens.append(SHORT_CASTLE)
    elif board.is_queenside_castling(move):
        tokens.append(LONG_CASTLE)
    else:
        # No special token is used to indicate a promotion - we just transform the pawn to a new piece
        if move.promotion is not None:
            tokens.append(encode_piece_map[move.promotion])
        else:
            # Piece
            tokens.append(encode_piece_map[board.piece_at(move.from_square).piece_type])

        if board.is_capture(move):
            tokens.append(TAKE)
        # Position
        tokens.append(FIRST_POSITION + move.to_square)

    if board.is_into_check(move):
        tokens.append(CHECK)
    return tokens


if __name__ == "main":
    with open("../../../rust/chessai/tokens.bin", "rb") as f:
        game = f.readline().strip()
        print(decode(game))
