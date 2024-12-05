from enum import Enum
from dataclasses import dataclass
from typing import List


@dataclass
class Ply:
    piece: str
    from_file: str
    from_rank: str
    file: str
    rank: str
    take: bool
    check: bool
    short_castle: bool
    long_castle: bool
    promotion_to: str
    checkmate: bool
    analysis: int

    def valid(self) -> bool:
        return self.piece != 'x' or self.short_castle or self.long_castle

    @staticmethod
    def get_analysis_str(analysis: int) -> str:
        return {
            -4: "??",
            -2: "?",
            -1: "?!",
            2: "!",
            4: "!!"
        }.get(analysis, "")

    def __str__(self) -> str:
        builder = []
        if self.short_castle:
            builder.append("O-O")
        elif self.long_castle:
            builder.append("O-O-O")
        else:
            if self.piece != 'P':
                builder.append(self.piece)
            if self.from_file != 'x':
                builder.append(self.from_file)
            if self.from_rank != 'x':
                builder.append(self.from_rank)
            if self.take:
                builder.append("x")
            builder.append(self.file)
            builder.append(self.rank)
            if self.promotion_to != 'x':
                builder.append("=")
                builder.append(self.promotion_to)

        if self.check:
            builder.append("+")
        if self.checkmate:
            builder.append("#")
        if self.analysis != 0:
            builder.append(self.get_analysis_str(self.analysis))

        return "".join(builder)


@dataclass
class Move:
    num: int
    white: Ply
    black: Ply = None

    def __str__(self) -> str:
        black_str = f" {self.black}" if self.black and self.black.valid() else ""
        return f"{self.num}. {self.white}{black_str}"


class Winner(Enum):
    NONE = "*"
    WHITE = "1-0"
    BLACK = "0-1"
    DRAW = "1/2-1/2"

    def __str__(self):
        return self.value


@dataclass
class Game:
    moves: List[Move]
    winner: Winner

    def __str__(self) -> str:
        builder = []
        for move in self.moves:
            builder.append(str(move))
            builder.append("  ")
        builder.append(str(self.winner))
        return "".join(builder)
