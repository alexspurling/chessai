import gzip


dataset_file = "/home/alex/Downloads/stockfish_dataset.csv"
games_file = "../stockfishgames2.gz"
tokens_file = "../stockfishgames_tokens.gz"


def extract_games():

    line_count = 0
    game_count = 0

    with open(dataset_file, 'rb') as f:
        with gzip.open(games_file, 'w') as w:
            for line in f:
                line_count += 1

                if line[:2] == b'1.':
                    end_pos = line.find(b'"')
                    game = line[0:end_pos]
                    # print(game)
                    w.write(game)
                    w.write(b'\n')

                    game_count += 1
                if line_count % 100000 == 0:
                    print("Line count", line_count, "Game count", game_count)


def build_char_set():

    all_chars = set()
    line_count = 0

    chars = ""

    with gzip.open(games_file, 'rb') as f:
        for line in f:
            line_count += 1

            for b in line:
                if b not in all_chars:
                    all_chars.add(b)
                    chars = ''.join([chr(i) for i in sorted(list(all_chars))])
                    print("Chars", chars)

            if line_count % 100000 == 0:
                print("Line count", line_count)

    return chars


# print("Extracting games into .gz file")
# extract_games()

# print("Counting unique characters")
# print(build_char_set())
# print("Done")

chars = "\n #+-.0123456789=BKNOQRabcdefghx"

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(str_data):
    return [stoi[c] for c in str_data]


def decode(tokens_list):
    return "".join([itos[i] for i in tokens_list])


# print("Tokenising games")
#
# with gzip.open(games_file, 'rt') as f:
#     with open(tokens_file, 'wb') as w:
#         line_count = 0
#         for line in f:
#             line_count += 1
#             w.write(bytes(encode(line)))
#
#             if line_count % 100000 == 0:
#                 print("Line count", line_count)
