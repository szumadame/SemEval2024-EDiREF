# def _tokenize(text):
#     return text.split()
#
#
# def _extract_relevant_data_lstm(dataset):
#     # Extract all utterances and emotions
#     utterance_list = []
#     emotions_list = []
#     for dialogue in dataset:
#         utterance_list = utterance_list + dialogue["utterances"]
#         emotions_list = emotions_list + dialogue["emotions"]
#
#     # Tokenize utterances and remove punctuation
#     tokenized_utterances = [_tokenize(utt.lower()) for utt in utterance_list]
#     tokenized_utterances = [[word.strip(string.punctuation) for word in sublist] for sublist in tokenized_utterances]
#
#     # Encode emotions
#     encoded_emotions = [EMOTIONS_ERC[emotion] for emotion in emotions_list]
#     return encoded_emotions, tokenized_utterances
#
#
# def _create_dataloader_lstm(train_dataset_path, test_dataset_path, batch_size):
#     with open(train_dataset_path) as f:
#         train_dataset = json.load(f)
#
#     with open(test_dataset_path) as f:
#         test_dataset = json.load(f)
#
#     train_encoded_emotions, train_tokenized_utterances = _extract_relevant_data_lstm(train_dataset)
#     test_encoded_emotions, test_tokenized_utterances = _extract_relevant_data_lstm(test_dataset)
#
#     # Build the vocabulary based on the training dataset
#     vocab = Counter(word for utterance in train_tokenized_utterances for word in utterance)
#
#     # Assign an index to each word in the vocabulary starting from 2
#     # We reserve '0' for padding and '1' for unknown words
#     word_to_index = {word: i + 2 for i, (word, _) in enumerate(vocab.items())}
#     vocab_size = len(word_to_index) + 2  # Including padding and unknown word tokens
#
#     # Convert the tokenized utterances to integer sequences
#     train_sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in train_tokenized_utterances]
#     test_sequences = [[word_to_index.get(word, 1) for word in utterance] for utterance in test_tokenized_utterances]
#
#     # Pad the sequences to have equal length
#     train_padded_sequences = pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True,
#                                           padding_value=0)
#     test_padded_sequences = pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True,
#                                          padding_value=0)
#
#     assert len(train_padded_sequences) == len(train_encoded_emotions)
#     assert len(test_padded_sequences) == len(test_encoded_emotions)
#
#     # Concatenate utterances with emotions
#     train_wrapped_dataset = DialogueDatasetWrapperForLSTM(data=train_padded_sequences, labels=train_encoded_emotions,
#                                                           vocab_size=vocab_size)
#     test_wrapped_dataset = DialogueDatasetWrapperForLSTM(data=test_padded_sequences, labels=test_encoded_emotions,
#                                                          vocab_size=vocab_size)
#
#     # Create weighted random sampler to counteract the imbalanced dataset
#     weighted_random_sampler = WeightedRandomSampler(weights=train_wrapped_dataset.class_weights,
#                                                     num_samples=len(train_wrapped_dataset))
#
#     # train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, sampler=weighted_random_sampler)
#
#     train_dataloader = DataLoader(train_wrapped_dataset, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_wrapped_dataset, batch_size=batch_size, shuffle=False)
#     return train_dataloader, test_dataloader