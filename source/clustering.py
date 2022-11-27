from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import matplotlib.pyplot as plt # for plotting§ 
from sklearn import metrics 
###

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def labelMap(vol, pred):
    cm = metrics.confusion_matrix(vol, pred)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.asarray(indexes)
    return indexes[1]
    
def convertLabels(lmap, pred):
    conv_preds = []
    for i in range(len(pred)):
        conv_preds.append(lmap[pred[i]])
    return np.array(conv_preds)

def plot_confusion_matrix(predictions, input_data, input_labels, classes):
    
    # Compute the confusion matrix by comparing the test labels (ds.test_labels) with the test predictions
    cm = metrics.confusion_matrix(input_labels, predictions, labels=[0, 1, 2, 3])
    cm = cm.astype('float')

    # Normalize the confusion matrix results. 
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.matshow(cm_norm)

    plt.title('Confusion matrix', y=1.08)
    
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(classes)
    
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt), 
                    ha="center", va="center", 
                    color="white" if cm_norm[i, j] < thresh else "black")
    plt.show()