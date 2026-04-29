from sklearn.neural_network import MLPRegressor

X = df[["R"]]
y = df.filter(like="theta")

#model = MLPRegressor(hidden_layer_sizes=(64,64))
model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=3000,
    learning_rate_init=0.001,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

model.fit(X_scaled, y_scaled)
