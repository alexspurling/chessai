from model import ChessModel

m = ChessModel("stockfishmodel.pt", "cuda")
m.train("tokens_stockfish.bin")

