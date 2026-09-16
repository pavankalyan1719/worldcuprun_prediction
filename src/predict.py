def predict_score(model, sample):

    prediction = model.predict(sample)[0]

    return prediction