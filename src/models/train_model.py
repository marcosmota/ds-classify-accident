
from typing import Dict
from pyparsing import Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data.dataset import Dataset

class Trainer:
    def __init__(self, models: Dict[str, Any], preprocessor: Any):
        self._models = models
        self._preprocessor = preprocessor
        self._results = {}

    def train(self, dataset: Dataset):
        for name, classifier in self._models.items():
            self._results[name] = self._train(classifier, dataset)

    def _train(self, classifier, dataset: Dataset):
        X_train, y_train = dataset.training

        pipeline = Pipeline(steps=[
            ('preprocessor', self._preprocessor),
            ('classifier', classifier)
        ])
        pipeline.fit(X_train, y_train)

        X_test, y_test = dataset.testing

        validation_score = pipeline.score(X_test, y_test)

        tree_ds_crossval = cross_val_score(pipeline, X_test, y_test)
        y_pred = pipeline.predict(X_test)
        predictions = pipeline.predict_proba(X_test)[:, 1]
        confusion = confusion_matrix(y_test, y_pred)

        fpr_model, tpr_model, _ = roc_curve(y_test, predictions)
        auc_model = auc(fpr_model, tpr_model)
        return {
            "pipeline": pipeline,
            "validation_score": validation_score,
            "cross_validation_score": tree_ds_crossval.mean(),
            "confusion_matrix": confusion,
            "predictions": predictions,
            "fpr_model": fpr_model,
            "tpr_model": tpr_model,
            "auc_model" :auc_model
        }
    
    def get_model(self, name):
        return self._results[name]["pipeline"]
    
    def get_best_model(self):
        sorted_results = sorted(self._results.items(), key=lambda x: x[1]['cross_validation_score'], reverse=True)
        return sorted_results[0][1]['pipeline']

    def plot_result_model(self, name):
        result = self._results[name]

        
        print("{} Score: {:.2f}".format(name, result['validation_score']))
        print("{} CV Score: {:.2f}".format(name, result['cross_validation_score']))
   
        confusion = result["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(8,6), dpi=100)

        disp = ConfusionMatrixDisplay(confusion, display_labels=["Sim", "Nao"]) 
        plt.title('Matrix Confusion {}'.format(name))

        ax.set(title='Matrix de Confusão para {}'.format(name))
        disp.plot(ax=ax)
        

    def plot_roc_auc(self):
        plt.figure()
        for name, result in self._results.items():            
            fpr_model = result['fpr_model']
            tpr_model = result['tpr_model']
            auc_model = result['auc_model']

            plt.plot(fpr_model, tpr_model, label='{} (AUC = {:.2f})'.format(name, auc_model))

        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de FP')
        plt.ylabel('Taxa de TP')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.show()


    def plot_results(self):
        print("------------------- Resultados ----------------------------")
        for name, _ in self._results.items():
            self.plot_result_model(name)

        
        
        
        
    
