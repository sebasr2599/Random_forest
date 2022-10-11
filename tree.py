from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

columns = ["white_king_file", "white_king_rank", "white_rook_file", "white_rook_rank","black_king_file", "black_king_rank","classes"]

df = pd.read_csv('krkopt.data', names=columns)
print("Data")
print(df.head())

print("shuffled")
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

#Label encoding
le_wrf = LabelEncoder()
le_wkf = LabelEncoder()
le_bkf = LabelEncoder()

le_classes = LabelEncoder()

df.white_rook_file = le_wrf.fit_transform(df.white_rook_file)
df.white_king_file = le_wkf.fit_transform(df.white_king_file)
df.black_king_file = le_bkf.fit_transform(df.black_king_file)

df.classes = le_classes.fit_transform(df.classes)
print("Label encoded")
print(df.head())

# Divide into x and y
df_y = df["classes"]
del df["classes"]
columns.remove("classes")
df_x = df.copy()

print(f'x:\n {df_x.head()}\n y:\n {df_y.head()}')

# test, train x 
df_x_train = df_x[:27000]
df_x_test = df_x[27000:]

# test and train y 
df_y_train = df_y[:27000]
df_y_test = df_y[27000:]

d_tree = DecisionTreeClassifier(max_depth = 8)
d_tree.fit(df_x_train,df_y_train)
             
y_prediction = d_tree.predict(df_x_test)


# accuracy_score(df_x_test, y_prediction)
pred = d_tree.predict(df_x_test.iloc[[0]])
print(f'To predict: {df_x_test.iloc[[0]]}')
print(f'Real: {df_y_test.iloc[[0]]}')
print(f'Prediction: {pred}')

print(le_classes.classes_)
print(le_classes.inverse_transform(pred))
in_list = []
in_list.append(input("Enter the white king file "))
in_list.append(int(input("Enter the white king rank ")))
in_list.append(input("Enter the rook king file "))
in_list.append(int(input("Enter the rook king rank ")))

in_list.append(input("Enter the black king file "))
in_list.append(int(input("Enter the black king rank ")))

in_list[0] = le_wkf.transform([in_list[0]])
in_list[2] = le_wrf.transform([in_list[2]])
in_list[4] = le_bkf.transform([in_list[4]])

in_list[0] = in_list[0][0]
in_list[2] = in_list[2][0]
in_list[4] = in_list[4][0]


print(f'{in_list[0]} {in_list[2]} {in_list[4]}')
print(df_x_test.iloc[[0]])
print(in_list)

# np_array = np.array(in_list)
np_array = np.array([[0, 2, 1, 5, 7, 7]])
pred_df = pd.DataFrame(np_array,columns=columns)
print(pred_df.info())
print(pred_df)

in_pred = d_tree.predict(pred_df)
print(f'Prediction: {in_pred}')
print(le_classes.inverse_transform(in_pred))
# in_pred = d_tree.predict(in_white_king_file, in_white_king_rank, in_white_rook_file, in_white_rook_rank, in_black_king_file, in_black_king_rank)

"""
samples: 28056
shuffle
divide into x and y
divide the x and y in train and test
train model
check error %
display tree
predict on data
end
"""

