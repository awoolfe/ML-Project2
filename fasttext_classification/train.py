import fasttext
model = fasttext.train_supervised('../data/fasttext.train.processed.txt')
print(model.words)
print(model.labels)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test('../data/fasttext.valid.processed.txt'))