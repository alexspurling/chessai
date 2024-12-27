from model import ChessModel, Medium, Small

m = ChessModel("stockfishmodel_chars2.pt", vocab_size=44, params=Medium, device="cuda")
m.train("stockfishgames2.gz")

