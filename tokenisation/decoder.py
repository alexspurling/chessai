

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


def decode_position(b):
    val = b - FIRST_POSITION
    file = chr(ord('a') + (val % 8))
    rank = int(val / 8) + 1
    return f"{file}{rank}"


def decode(token_data: []) -> str:

    output = []
    for b in token_data:
        if FIRST_NUM <= b <= LAST_NUM:
            output.append(f"{b - FIRST_NUM}.")
        elif b == SHORT_CASTLE:
            output.append(f"O-O")
        elif b == LONG_CASTLE:
            output.append(f"O-O-O")
        elif b == PAWN:
            output.append(f"P")
        elif b == BISHOP:
            output.append(f"B")
        elif b == KNIGHT:
            output.append(f"N")
        elif b == QUEEN:
            output.append(f"Q")
        elif b == ROOK:
            output.append(f"R")
        elif b == KING:
            output.append(f"K")
        elif b == TAKE:
            output.append(f"x")
        elif b == CHECK:
            output.append(f"+")
        elif b >= FIRST_POSITION:
            output.append(decode_position(b))
        else:
            output.append(f"{b}")

    return " ".join(output)


if __name__ == "main":
    with open("../../../rust/chessai/tokens.bin", "rb") as f:
        game = f.readline().strip()
        print(decode(game))
