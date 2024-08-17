import pickle
import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from xgboost import XGBClassifier
from collections import Counter
from tqdm import tqdm

from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.decomposition import *

from rdkit import Chem
from rdkit.Chem import AllChem

from scipy.sparse import csr_matrix

def train_models_classification(models, X, y, kfold_splits=10):
        """
        Trains multiple classification models using k-fold cross-validation and evaluates their performance.
    
        Args:
            models (list of tuples): A list of tuples where each tuple contains a model name (str) and 
                                     the model object (scikit-learn estimator).
            X (pd.DataFrame): The input features for training the models.
            y (pd.Series): The target labels for classification.
            kfold_splits (int, optional): The number of splits for cross-validation. Default is 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the average performance metrics for each model, including:
                          - 'Model': The name of the model.
                          - 'Accuracy': The average accuracy across all folds.
                          - 'Sensitivity': The average recall (sensitivity) across all folds.
                          - 'Specificity': The average specificity across all folds.
                          - 'Precision': The average precision across all folds.
                          - 'F1': The average F1 score across all folds.
        """
        results_list = []
        columns = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
    
        kfold = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    
        for name, model in models:
            scores = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'precision': [], 'f1': []}
    
            for train_index, test_index in kfold.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
    
                scores['accuracy'].append(accuracy_score(y_test, y_pred))
                scores['sensitivity'].append(recall_score(y_test, y_pred, zero_division=1))
                scores['specificity'].append(recall_score(y_test, y_pred, pos_label=0, zero_division=1))
                scores['precision'].append(precision_score(y_test, y_pred, zero_division=1))
                scores['f1'].append(f1_score(y_test, y_pred, zero_division=1))

            avg_scores = {metric: np.mean(scores_list) for metric, scores_list in scores.items()}
    
            results_list.append([name, *avg_scores.values()])

        results_df = pd.DataFrame(results_list, columns=columns)
        return results_df
    
def validate_models_classification(self, models, X, y, X_val, y_val, kfold_splits=10):
        """
        Trains multiple classification models on the provided training data and evaluates
        them on validation data.
    
        Args:
            models (list of tuples): A list of tuples where each tuple contains a model name (str) and 
                                     the model object (scikit-learn estimator).
            X (pd.DataFrame): The input features for training the models.
            y (pd.Series): The target labels for training the models.
            X_val (pd.DataFrame): The input features for validation.
            y_val (pd.Series): The target labels for validation.
            kfold_splits (int, optional): The number of splits for cross-validation. Default is 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the performance metrics for each model
            on the validation set, including:
                          - 'Model': The name of the model.
                          - 'Accuracy': The accuracy on the validation set.
                          - 'Sensitivity': The recall (sensitivity) on the validation set.
                          - 'Specificity': The specificity on the validation set.
                          - 'Precision': The precision on the validation set.
                          - 'F1': The F1 score on the validation set.
        """
        results_list = []
        columns = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
    
        for name, model in models:
            model.fit(X, y)
            y_pred = model.predict(X_val)
    
            accuracy = accuracy_score(y_val, y_pred)
            sensitivity = recall_score(y_val, y_pred, zero_division=1)
            specificity = recall_score(y_val, y_pred, pos_label=0, zero_division=1)
            precision = precision_score(y_val, y_pred, zero_division=1)
            f1_measure = f1_score(y_val, y_pred, zero_division=1)
    
            results_list.append([name, accuracy, sensitivity, specificity, precision, f1_measure])
            results_df = pd.DataFrame(results_list, columns=columns)
    
        return results_df
      
class DecisionTreeHelper:
    def __init__(self, model, train_dtf_activity, smiles, feature_importance):
        self.model = model
        self.train = train_dtf_activity[feature_importance.Feature.head(26)]
        self.activity = train_dtf_activity.iloc[:,-1]
        self.smiles = smiles
        self.n_nodes = self.model.tree_.node_count
        self.children_left = self.model.tree_.children_left
        self.children_right = self.model.tree_.children_right
        self.tree = self.model.tree_
        self.leaf_nodes = [node for node in range(self.n_nodes) if self.children_left[node] == self.children_right[node] == -1]
        self.feature_names = feature_importance["Feature"].tolist()
        self.dec_path = None
        self.leaf_id = model.apply(self.train)
        self.dense_matrix = None

    def decision_paths(self, show_leafs=True):
        self.dec_path = self.model.decision_path(self.train, check_input=True)
        dense_matrix = self.dec_path.toarray()
        dense_matrix_df = pd.DataFrame(dense_matrix)
        
        if not show_leafs:
            return dense_matrix_df
        
        def leaf_or_node(col):
            return f"leaf_{col}" if col in self.leaf_nodes else f"node_{col}"
        
        dense_matrix_df.columns = [leaf_or_node(col) for col in dense_matrix_df.columns]
        
        self.dense_matrix = pd.concat([dense_matrix_df,self.activity], axis = 1)
                
        return dense_matrix_df
    
    def find_leaves(self, return_list=True):
        dense_matrix = self.dec_path.toarray()
        related_node_list = []
        
        for path in dense_matrix:
            leaf_node = None
            for node in self.leaf_nodes:
                if path[node] == 1:
                    leaf_node = node
                    related_node_list.append(leaf_node)
                    break
        
        leaves_df = pd.DataFrame({
            'sample': self.smiles,
            'leaf': related_node_list
        })
        
        if return_list:
            return leaves_df
        else:
            select = int(input("What is the index of the sample whose leaf number you want to know? "))
            print(f"The compound number {select}")
            print(f"whose smiles is {leaves_df.iloc[select]['sample']}")
            print(f"is in the leaf number {leaves_df.iloc[select]['leaf']}")

    def find_leaf_parent(self, leaf_node):
        parent_nodes = {i: (self.children_left[i], self.children_right[i]) for i in range(self.n_nodes) if self.children_left[i] != -1 or self.children_right[i] != -1}
        
        for parent, (left, right) in parent_nodes.items():
            if left == leaf_node or right == leaf_node:
                return parent, left, right
        return None, "NaN", "NaN"
    
    def find_splits(self, pure_node = True):
        parent_leaves = []
        right_leaves = []
        left_leaves = []
        origin_leaves = []
        
        for leaf in self.leaf_nodes:
            first_parent_node, left_child, right_child = self.find_leaf_parent(leaf)
            if first_parent_node is not None:
                parent_leaves.append(first_parent_node)
                right_leaves.append(right_child)
                left_leaves.append(left_child)
                
        df_parent_child = pd.DataFrame({'left_leaves': left_leaves,
                                        'parent_leaves': parent_leaves,
                                        'right_leaves': right_leaves})                
        if pure_node == True:

            df_parent_child = df_parent_child[
                df_parent_child['right_leaves'].isin(self.leaf_nodes) & 
                df_parent_child['left_leaves'].isin(self.leaf_nodes)
            ]

            df_parent_child = df_parent_child.drop_duplicates(subset=['right_leaves'])
            pure_nodes = df_parent_child.reset_index(drop=True)
            
            return pure_nodes
        
        elif pure_node == False:
            
            return df_parent_child

    def describe(self, pure_nodes):
        original_leaf = []
        connected_leaf = []
        leaf_incommon = []
        count_original = []
        count_original_value = []
        count_connected_leaf = []
        count_connected_value = []
    
        leaves_list = list(pure_nodes['left_leaves'].value_counts().index)
    
        for leaf in leaves_list:
            leaf_col_name = f'leaf_{leaf}'
            leaf_data = self.dense_matrix[self.dense_matrix[leaf_col_name] == 1].iloc[:, -1]
    
            if leaf_data.empty:
                continue
            
            value_counts = leaf_data.value_counts()
            leaf_1_bool = value_counts.index[0]
            leaf_1_value = value_counts.values[0]
    
            if leaf_1_bool == 1:
                leaf_other_row = pure_nodes[pure_nodes['left_leaves'] == leaf]
                if leaf_other_row.empty:
                    continue
    
                leaf_other = leaf_other_row['right_leaves'].values[0]
                leaf_common = leaf_other_row['parent_leaves'].values[0]
                
                leaf_other_data = self.dense_matrix[self.dense_matrix[f'leaf_{leaf_other}'] == 1].iloc[:, -1]
                if leaf_other_data.empty:
                    continue
    
                value_counts_other = leaf_other_data.value_counts()
                leaf_2_bool = value_counts_other.index[0]
                leaf_2_value = value_counts_other.values[0]
    
                if leaf_1_bool != leaf_2_bool:
                    print(f"Node {leaf} is connected to node {leaf_other} by the common node {leaf_common}")
                    print(f"Node {leaf} has {leaf_1_value} {'active' if leaf_1_bool == 1 else 'inactive'} compound(s)")
                    print(f"Node {leaf_common} has {leaf_2_value} {'active' if leaf_2_bool == 1 else 'inactive'} compound(s)")
                    print("----------")
    
                    original_leaf.append(leaf)
                    connected_leaf.append(leaf_other)
                    leaf_incommon.append(leaf_common)
                    count_original.append(leaf_1_value)
                    count_original_value.append(leaf_1_bool)
                    count_connected_leaf.append(leaf_2_value)
                    count_connected_value.append(leaf_2_bool)
    
        leaf_descr = pd.DataFrame({
            'leaf': original_leaf,
            'leaf_conn': connected_leaf,
            'leaf_common': leaf_incommon,
            'leaf_count': count_original,
            'leaf_value': count_original_value,
            'leaf_conn_count': count_connected_leaf,
            'leaf_conn_value': count_connected_value
        })
    
        return leaf_descr

    def backward_search(self, leaf_node):
        
        parent_nodes = {}
        
        for i in range(self.n_nodes):
            if self.children_left[i] != -1:
                parent_nodes[self.children_left[i]] = i
            if self.children_right[i] != -1:
                parent_nodes[self.children_right[i]] = i
        
        current_node = leaf_node
        ancestor_nodes = []
        
        while current_node in parent_nodes:
            parent_node = parent_nodes[current_node]
            ancestor_nodes.append(parent_node)
            current_node = parent_node
        
        print(f'The nodes from which node {leaf_node} is derived are, backwards: {ancestor_nodes}')
        
        print("Below is the list of nodes resulting from the split of the listed nodes")
        
        for parent in ancestor_nodes:
            left_child = self.children_left[parent]
            right_child = self.children_right[parent]
            print(f'Node {parent} is splitted in: left_child = {left_child}, right_child = {right_child}')
            
    def node_split_criteria(self, node_id):
        
        is_leaf = self.children_left[node_id] == self.children_right[node_id]
        
        if is_leaf:
            print(f"Node {node_id} is a leaf node, there is not slitting criteria")
        else:
            feature_index = self.tree.feature[node_id]
            threshold_value = self.tree.threshold[node_id]
            print(f"The node {node_id} slitting criteria is: Feature: {self.feature_names[feature_index]}, threshold = {threshold_value}")
            print(f"The node splits into left node {self.children_left[node_id]} if {self.feature_names[feature_index]} ≤ {threshold_value}")
            print(f"and in right node {self.children_right[node_id]} if {self.feature_names[feature_index]} > {threshold_value}")
            
    def sample_decision_path(self, sample_id):

        node_index = self.dec_path.indices[
            self.dec_path.indptr[sample_id]: self.dec_path.indptr[sample_id + 1]
        ]

        print(f"Rules followed by sample number {sample_id}:\n")
        for node_id in node_index:

            if self.leaf_id[sample_id] == node_id:
                continue

            node_feature = self.tree.feature[node_id]
            node_threshold = self.tree.threshold[node_id]

            sample_feature_value = self.train.iloc[sample_id, node_feature] 

            if sample_feature_value <= node_threshold:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print(
                f"Decision node {node_id}: {self.train.columns[node_feature]} = {sample_feature_value} "
                f"{threshold_sign} {node_threshold}"
            )
            
        final_value = self.dense_matrix[self.dense_matrix[f'leaf_{self.leaf_id[sample_id]}']==1].iloc[:,-1].value_counts().index[0]
        print("------------------------")    
        print(f"Sample {sample_id} ends up in node {self.leaf_id[sample_id]}")
        print(f"and it is classified as {'active' if final_value == 1 else 'inactive'}")
        print("------------------------")
        
    def find_node_by_feat(self, feature_to_find = None):
        if feature_to_find is None:
            feature_to_find = self.feature_names[0]
            
        node_ids = []
        splitting_criteria = []
        threshold_value_list = []
    
        for node_id in range(self.n_nodes):
            is_leaf = self.children_left[node_id] == self.children_right[node_id]
            
            if is_leaf:
                splitting_criteria.append("no_leaf")
            else:
                feature_index = self.tree.feature[node_id]
                threshold_value = self.tree.threshold[node_id]
                feature_name = self.feature_names[feature_index]
                splitting_criteria.append(feature_name)
                threshold_value_list.append(threshold_value)
                        
            node_ids.append(node_id)
        
        df_criteria = pd.DataFrame({
            'splitting_criteria': splitting_criteria,
        })
        
        row_index = df_criteria[df_criteria["splitting_criteria"] == feature_to_find].index

        print(f"Descriptor '{feature_to_find}' is the decision criteria of node(s): {row_index.tolist()}")