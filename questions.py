import nltk
import os
import sys
import string
from collections import Counter

FILE_MATCHES = 3
SENTENCE_MATCHES = 3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    load_dict = {}
    load = os.listdir(directory)
    for file in load:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as text:
            load_dict[file] = text.read()

    return load_dict

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    words = [word.lower() for word in nltk.word_tokenize(document) if word not in string.punctuation
             and word not in nltk.corpus.stopwords.words("english")]

    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    import math

    N = len(documents)
    idf_dict = {}
    for doc, words in documents.items():
        unique_words = set(words)
        for word in unique_words:
            if word not in idf_dict:
                val = 0
                for passage in documents.values():
                    if word in passage:
                        val += 1
                if val != 0:
                    idf_dict[word] = math.log(N/val)

    return idf_dict

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    from collections import Counter
    tf = dict.fromkeys(query, 0)
    tf_idf = {}

    """for doc, words in files.items():
        freq = Counter(words)
        for item in query:
            for word in freq:
                if item == word:
                    tf[item] = freq[word]
            tf_idf[doc] += tf[item] * idfs[item] if item in idfs else 0"""

    for doc, words in files.items():
        tf_idf[doc] = 0
        unique_words = set(words)
        for word in unique_words:
            if word in query:
                tf[word] = words.count(word)
                tf_idf[doc] += tf[word] * idfs[word] if word in idfs else 0


    top_files = []
    for count in Counter(tf_idf).most_common(n):
        top_files.append(count[0])

    return top_files



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sent_score = {} #dictionary mapping a sentence to it's matching word measure and query term density as a tuple
    top_sentences = []
    for sentence, words in sentences.items():
        sent_idf = float()
        count = int()
        unique_words = set(words)
        for word in unique_words:
            if word in query:
                count += sentence.count(word)
                sent_idf += idfs[word]
        term_density = count / len(words)
        sent_score[sentence] = (sent_idf, term_density)

    for count in Counter(sent_score).most_common(n):
        top_sentences.append(count[0])

    return top_sentences

if __name__ == "__main__":
    main()
