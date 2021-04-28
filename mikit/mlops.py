from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm


class WrapperMethod():
    def __init__(self, X, y, feature_names, train_rate=0.9):
        """
        Args:
            X(numpy) : Features
            y(numpy, list) : Objective variable
            feature_names(numpy, list) : Feature names
            train_rate(float) : Percentage of training data used in grid search
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0, train_size=train_rate)
        self.feature_names = feature_names
    
    def forward_search(self, clf, best_score, best_params_idx):
        """
        Args:
            clf(sklearn) : Learning model
            best_score(float) : R2 score of CV
            best_params_idx(list) : Index of adopted features
        Return:
            best_params_idx(list) :  Index of adopted features
            cv_score(float) : R2 score of CV
            train_score(float) : R2 score of training
            test_score(float) :  R2 score of testing
            clf.best_estimator_(sklearn) : Best learning model
        """
        for f in tqdm(range(len(self.feature_names))):
            if f not in best_params_idx:
                new = [f] + best_params_idx
                clf.fit(self.X_train[:, new], self.y_train)
                if clf.best_score_ > best_score:
                    best_score = clf.best_score_
                    best_param_idx = [f]
        try:
            best_params_idx = best_params_idx + best_param_idx
            cv_score = best_score
            clf.fit(self.X_train[:, best_params_idx], self.y_train)
            y_train_predict = clf.predict(self.X_train[:, best_params_idx])
            y_test_predict = clf.predict(self.X_test[:, best_params_idx])
            train_score = r2_score(self.y_train, y_train_predict)
            test_score = r2_score(self.y_test, y_test_predict)
        except:
            best_params_idx, cv_score, train_score, test_score = None, None, None, None
        return best_params_idx, cv_score, train_score, test_score, clf.best_estimator_
    
    def calc_forward(self, model, hyper_params):
        """
        Args:
            model(sklearn) : Learning model
            hyper_params(dict) : hyper parameters
        Return:
            best_params_idx(list) :  Index of adopted features
            scores(dict) : R2 scores of [cv, train, test]
            best_clf(sklearn) : Best learning model
        """
        cv_score, new_cv_score = 0.01, 0
        # best_params_idx, cv_scores, train_scores, test_scores = list(), list(), list(), list()
        best_params_idx, scores = list(), dict()
        scores["cv"], scores["train"], scores["test"] = list(), list(), list()
        clf = GridSearchCV(model, hyper_params)
        while cv_score != None:
            new_cv_score = cv_score
            new_best_params_idx, cv_score, train_score, test_score, new_best_clf = self.forward_search(clf, cv_score, best_params_idx)
            if cv_score != None:
                best_params_idx = new_best_params_idx
                best_clf = new_best_clf
                scores["cv"].append(cv_score)
                scores["train"].append(train_score)
                scores["test"].append(test_score)
        return best_params_idx, scores, best_clf



