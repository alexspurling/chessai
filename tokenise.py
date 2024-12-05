import gzip
import time

from tokenisation.pgnparser import parse

parsed = 0

total_start_time = time.time()
total_token_time = 0

#############
# This python implementation of the parser is slow. Use the rust version to produce tokens quickly
#############

with gzip.open("bestgames3.gz", mode="rt", encoding="utf-8") as file:
    i = 0
    for line in file:
        pgn_game = line.strip()

        start_time = time.time()
        game = parse(pgn_game)
        if game:
            parsed += 1
        # if tokenize_regex(game):
        #     parsed += 1
        total_token_time += (time.time() - start_time)

        i += 1
        if i % 10000 == 0:
            print(f"Parsed {parsed} out of {i} games.")
        if i > 100000:
            break

total_time = (time.time() - total_start_time)
print(f"Total time taken: {total_time}, token time: {total_token_time}, parsed: {parsed}")
