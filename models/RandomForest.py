from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics

class RandomForest:
    def run_RF(X,y,split):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split 
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=split)
        clf_rf = RandomForestClassifier(n_estimators=100, max_features='log2')
        clf_rf = clf_rf.fit(X_train, y_train)
        
        y_pred = clf_rf.predict(X_test)
        print("Classification Report")
        print(classification_report(y_test,y_pred))
        print("Accuracy score for Random Forest:",accuracy_score(y_test,y_pred))
        scores = cross_val_score(clf_rf, X_train, y_train, cv=10)
        print("Calculating the cross validated score for the classifier")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return