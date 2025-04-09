from ml import load_data, BertModel

file = "../data/training.1600000.processed.noemoticon.csv"
x_train, x_test, x_val, y_train, y_test, y_val = load_data(file)
model = BertModel(x_train, y_train)
model.train()