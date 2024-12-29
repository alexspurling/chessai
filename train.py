from model import ChessModel, Medium, Small

m = ChessModel("stockfishmodel_chars3.pt", vocab_size=32, params=Small, device="cpu")
m.train("stockfishgames_tokens.bin")

