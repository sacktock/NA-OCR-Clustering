import os
import argparse
import re

infile = None
outfile = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", default=None)
    parser.add_argument("-o", "--output-file", default=None)
    args = parser.parse_args()

    infile = str(args.input_file)
    outfile = str(args.output_file)

    assert infile, "No input file provided"
    assert outfile, "No output file provided"

    with open(infile) as openfile:
        text = openfile.read()

    import re

    # Remove all words that contain a non-letter character
    text = re.sub(r"\b\w*[^\W\d_]+\w*\b", lambda m: m.group() if all(c.isalpha() for c in m.group()) else "", text)

    # Remove all characters that aren't letters or whitespace
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove all completely uppercase words
    text = re.sub(r"\b[A-Z]+\b", "", text)

    # Remove all words with length less than or equal to 2
    text = re.sub(r"\b\w{1,2}\b", "", text)

    # Replace all whitespace characters with a space
    text = re.sub(r"\s+", " ", text)

    with open(outfile, "w") as openfile:
        openfile.write(text)

    print("Done")
