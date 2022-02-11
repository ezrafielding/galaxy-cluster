from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from zoobot import label_metadata, schemas
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment as linear_assignment
import time

def findChoice(frac):
    choice = np.zeros_like(frac)
    choice[np.arange(len(frac)), frac.argmax(1)] = 1
    return choice

def getQuestionClasses(auto_f, volunteers, question, seed):
    qcol_name = question.text+'_total-votes'
    fcol_names = [(cols.text+'_fraction') for cols in question.answers]
    anscol_names = [cols.text for cols in question.answers]
    valid_feats = []
    
    valid_vol = volunteers.query('`{}`/`smooth-or-featured_total-votes` >= 0.5'.format(qcol_name))
    valid_idx = valid_vol.index.tolist()
    vol_results = valid_vol[fcol_names].values
    
    auto_values = auto_f.values
    
    for i in valid_idx:
        valid_feats.append(auto_values[i])
        
    rounded_vol_results = findChoice(np.asarray(vol_results))
    support = len(rounded_vol_results)
    
    pred_results = AgglomerativeClustering(n_clusters=len(fcol_names)).fit_predict(valid_feats)
    
    vol_classes = np.argmax(rounded_vol_results, axis=1)
    
    return valid_idx, support, anscol_names, np.array(pred_results), np.array(vol_classes)

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def labelMap(vol, pred):
    cm = confusion_matrix(vol, pred)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.asarray(indexes)
    return indexes[1]
    
def convertLabels(lmap, pred):
    conv_preds = []
    for i in range(len(pred)):
        conv_preds.append(lmap[pred[i]])
    return np.array(conv_preds)

auto_features = pd.read_csv("/users/ezraf/galaxyDECaLS/autoencoder/extracted_features.csv")
auto_features = auto_features.drop('file_loc',axis=1)
decals_test = pd.read_csv('/users/ezraf/galaxyDECaLS/Ilifu_data/decals_ilifu_test.csv')
schema = schemas.Schema(label_metadata.decals_pairs, label_metadata.get_gz2_and_decals_dependencies(label_metadata.decals_pairs))

total_report = {}
seeds = [6589,4598,2489,9434,7984,1238,6468,5165,3246,8646]
total_time = {}
for question in label_metadata.decals_pairs:
        total_report[question] = {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'support': 0
        }
for question in label_metadata.decals_pairs:
    total_time[question] = {}
    print('Starting Clustering for ',question)
    start = time.time()
    idxs, support, anscols, valid_preds, valid_vol = getQuestionClasses(auto_features, decals_test, schema.get_question(question), None)
    lmap = labelMap(valid_vol, valid_preds)
    conv_preds = convertLabels(lmap, valid_preds)
    question_report = precision_recall_fscore_support(y_pred=conv_preds, y_true=valid_vol, average='weighted')
    total_report[question]['precision'] += question_report[0]
    total_report[question]['recall'] += question_report[1]
    total_report[question]['f1'] += question_report[2]
    end = time.time()
    total_report[question]['support'] = support
    total_time[question]['total'] = end - start
    print('Question: ',question,' Completed 1 time')
    print('--------------------------------------------------------------')

report_df = pd.DataFrame.from_dict(total_report, orient='index')
time_df = pd.DataFrame.from_dict(total_time, orient='index')

report_df.to_csv("/users/ezraf/clusterResults/agg_accuracy.csv")
time_df.to_csv("/users/ezraf/clusterResults/agg_time.csv")