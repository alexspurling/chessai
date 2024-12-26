from model import ChessModel, Medium

m = ChessModel("stockfishmodel.pt", Medium, "cuda")
m.train("tokens_stockfish.bin")

