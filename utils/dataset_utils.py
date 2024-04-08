def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]
    return arrays


def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    sequences[0] = sequences[0][words_to_cut:]
    return sequences
