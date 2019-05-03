from pandas_ml import ConfusionMatrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,accuracy_score 
class NBModel:
    def run_NB(X, y, split):
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)


         
    #     print("Score for Naive Baye's:",model.score(X_train, y_train))
    #     print("Classification Report")
    #     print(classification_report(y_test,y_pred))  
        print("Accuracy score for Naive  baye's:",accuracy_score(y_test,y_pred))
        print(f1_score(y_test, y_pred, average="micro"))
        # confusion_matrix = ConfusionMatrix(y_test, y_pred)
        # print("Confusion matrix:\n%s" % confusion_matrix)
        return