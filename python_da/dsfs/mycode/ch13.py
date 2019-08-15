from typing import Set
import re


def tokenize(text: str) -> Set[str]:
    text = text.lower()  # standardize
    all_words = re.findall("[a-z0-9']+", text)  # extract words
    return set(all_words)   # remove duplicates


assert tokenize("Data science is science") == {"data", "science", "is", "science"}


# define class for training data
from typing import NamedTuple


class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (2 * self.k + self.spam_messages)
        p_token_ham = (ham + self.k) / (2 * self.k + self.ham_messages)
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0

        # Iterate through each word in the vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Otherwise add the log probability of _not_ seeing it,
            # i.e. log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


# Testing the model
messages = [Message("spam rules", is_spam = True),
            Message("ham rules", is_spam = False),
            Message("hello ham", is_spam = False)]
model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "rules", "ham", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "hello": 1, "rules": 1}

# now use real data
from io import BytesIO
import requests
import tarfile

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]
OUTPUT_DIR = "/Users/dag/github/learning/python_da/dsfs/mycode"

for filename in FILES:
    content = requests.get(f"{BASE_URL}/{filename}").content

    # wrap in-memory bytes to use them as a file
    fin = BytesIO(content)

    # Extract all files into output directory
    with tarfile.open(fileobj=fin, mode="r:bz2") as tf:
        tf.extractall(OUTPUT_DIR)

import glob, re
path = "/Users/dag/github/learning/python_da/dsfs/mycode/*/*"

data: List[Message] = []
# glob.glob returns every filname that matches the wildcarded path
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    with open(filename, errors="ignore") as email_file:
        for line in email_file:
            if  line.startswith("Subject:"):
                subject = line.lstrip("Subject:")
                data.append(Message(subject, is_spam))
                break

import random
from scratch.machine_learning import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier(k = 2)
model.train(train_messages)

from collections import Counter

predictions = [(message, model.predict(message.text))
               for message in test_messages]
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                            for message, spam_probability in predictions)

print(confusion_matrix)