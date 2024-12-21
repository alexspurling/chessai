from model import ChessModel

m = ChessModel("stockfishmodel.pt")
m.train("tokens_stockfish.bin")

