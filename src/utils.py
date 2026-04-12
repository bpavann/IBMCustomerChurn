import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
try:
    from src.exception import CustomException
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.exception import CustomException
    

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e)
    
def model_metrics(y_true, y_pred):
    try:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, precision, recall, f1
    except Exception as e:
        raise CustomException(e)

    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        model_names = []
        train_acc_list = []
        test_acc_list = []
        gap_list = []

        best_index = 0
        best_trained_model = None
        best_params = None

        for i, (model_name, model) in enumerate(models.items()):

            # Train model
            if model_name in params and params[model_name]:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                trained_model = gs.best_estimator_
                model_params = gs.best_params_
            else:
                trained_model = model
                trained_model.fit(X_train, y_train)
                model_params = trained_model.get_params()

            # Prediction
            train_pred = trained_model.predict(X_train)
            test_pred = trained_model.predict(X_test)

            # Accuracy
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            gap = abs(train_acc - test_acc)

            # store results
            model_names.append(model_name)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            gap_list.append(gap)

            # best model selection
            best_gap = gap_list[best_index]
            best_test_acc = test_acc_list[best_index]

            if (test_acc > best_test_acc) or \
               (test_acc == best_test_acc and gap < best_gap):

                best_index = i
                best_trained_model = trained_model
                best_params = model_params
                best_model_name = model_name

        return {
            "best_model_name": best_model_name,
            "best_model": best_trained_model,
            "best_model_name": model_names[best_index],
            "best_params": best_params,
            "best_train_accuracy": train_acc_list[best_index],
            "best_test_accuracy": test_acc_list[best_index],
            "best_gap": gap_list[best_index]
        }

    except Exception as e:
        raise CustomException(e)
