from sklearn.preprocessing import RobustScaler
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm

def split_dataset(df, target_col='activity', test_size=0.2):
    """
    Split a DataFrame into training and test sets, maintaining the distribution of the target.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    target_col : str, optional
        The name of the target column. Default is 'activity'.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The training and test DataFrames.
    """
    df_activity_1 = df[df[target_col] == 1]
    df_activity_0 = df[df[target_col] == 0]

    train_activity_1, test_activity_1 = train_test_split(df_activity_1, test_size=test_size, random_state=42)
    train_activity_0, test_activity_0 = train_test_split(df_activity_0, test_size=test_size, random_state=42)

    dtf_train = pd.concat([train_activity_1, train_activity_0])
    dtf_test = pd.concat([test_activity_1, test_activity_0])

    # Stampare il rapporto tra le attività nei set di addestramento e di test
    print("Training set: activity 1 = {:.2f}%, activity 0 = {:.2f}%, dimension = {}".format(
        100 * dtf_train[target_col].mean(), 100 * (1 - dtf_train[target_col].mean()), dtf_train.shape))
    print("Test set: activity 1 = {:.2f}%, activity 0 = {:.2f}%, dimension = {}".format(
        100 * dtf_test[target_col].mean(), 100 * (1 - dtf_test[target_col].mean()), dtf_test.shape))

    return dtf_train, dtf_test

def remove_constant_columns(df):
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(const_cols, axis=1, inplace=True)
    return df

def remove_semi_constant_columns(df, threshold=0.1):
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    non_const_cols = df.columns[selector.get_support(indices=True)]
    df = df[non_const_cols]
    return df

def remove_duplicate_columns(df):
    df = df.T.drop_duplicates().T
    return df

def remove_highly_correlated_columns(df, threshold=0.90):
    """
    Remove columns from DataFrame that have a correlation higher than the given threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from which to remove columns.
    threshold : float, optional
        Correlation threshold. Columns with correlation higher than this value will be removed. 
        Default is 0.90.

    Returns
    -------
    pd.DataFrame
        DataFrame with highly correlated columns removed.
    """
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(to_drop, axis=1)
    return df

def round_values(value):
    """
    This function is useful when, when downloading data
    from a database such as Reaxys,
    you have decimal 'a;b;c;d;e' values and want to
    round them off by turning them into integers.

    Parameters
    ----------
    value : str
        String of decimal values separated by ';'.

    Returns
    -------
    str
        String of rounded integer values separated by ';'.
    
    Example usage:
    df['column'] = df['column'].apply(round_values)
    
    """
    
    rounded_values = [str(round(float(val))) if '.' in val else val for val in value.split(';')]
    return '; '.join(rounded_values)

def terapeutic_window(row, lower_limit=600, upper_limit=850):
    """
    given a string of integer values, separated by ';',
    this function returns the maximum value in the defined
    therapeutic window that appears. If no value is present,
    it returns the absolute maximum.
    
    Parameters
    ----------
    row : str
        String of int values separated by ';'.
    lower_limit : int
        lower limit for terapeutic_window.
    upper_limit : int
        upper limit for terapeutic_window.

    Returns
    -------
    int
        lambda_max in terapeutic windows of absolute lambda_max.
    
    Example usage for a single string:
    result = terapeutic_window("600; 700; 800; 900", lower_limit=600, upper_limit=850)
    
    Example for a pandas column:
    df['column'] = df['column'].apply(terapeutic_window)
    
    """
    
    values = [int(val) for val in row.split(';')]
    valid_values = [val for val in values if lower_limit <= val <= upper_limit]
    if valid_values:
        return max(valid_values)
    else:
        return max(values)
    
def filter_lambda_range(df, min_value, max_value, column_name = "target"):
    """
    Filter the DataFrame based on the lambda_max column within the specified range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be filtered.
    column_name : str
        Column to be filtered.
        default is 'target'
    min_value : int or float
        Minimum value for the lambda_max column.
    max_value : int or float
        Maximum value for the lambda_max column.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    
    Example usage:
    df_filtered = filter_lambda_range(df, min_value=1, max_value=1000, column_name = "target")
    
    """
    return df[(df[column_name] >= min_value) & (df[column_name] <= max_value)]

class TargetAnalizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def continuous_features(self, column_name, n_bins=20, color_hex='#333333', alpha_value=0.7):
        """
        Generate histogram and statistical summary for the specified column.

        Parameters
        ----------
        column_name : str
            Name of the column to analyze.
        n_bins: int
            Number of bins.
            default is 20
        color_hex : str
            color hex code.
            default is #333333
        alpha_value : float
            trasparency
            default is 0.7

        Returns
        -------
        dict
            Dictionary containing statistical summary.
            
        Example usage:
        analyzer = TargetAnalizer(df)

        # description of selected column
        summary = analyzer.continuous_features('column', n_bins = 10, color_hex="#121245", alpha_value = 0.1)
        """
        column_data = self.dataframe[column_name]
        
        # Plot histogram
        plt.hist(column_data, bins=n_bins, color=color_hex, alpha=alpha_value)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title('Histogram of ' + column_name)
        
        # Plot lines for mean, median, and other metrics
        plt.axvline(column_data.mean(), color='green', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(column_data.median(), color='yellow', linestyle='dashed', linewidth=1, label='Median')
        plt.axvline(column_data.quantile(0.25), color='orange', linestyle='dashed', linewidth=1, label='25th Percentile')
        plt.axvline(column_data.quantile(0.75), color='purple', linestyle='dashed', linewidth=1, label='75th Percentile')
        
        plt.legend()  # Show legend
        
        plt.show()
        
        # Statistical summary
        print(f"Mean = {column_data.mean():.2f}")
        print(f"Median = {column_data.median():.2f}")
        print(f"Standard Deviation = {column_data.std():.2f}")
        print(f"Minimum = {column_data.min():.2f}")
        print(f"Maximum = {column_data.max():.2f}")
        print(f"25th Percentile = {column_data.quantile(0.25):.2f}")
        print(f"75th Percentile = {column_data.quantile(0.75):.2f}")
        print(f"Interquartile Range = {column_data.quantile(0.75) - column_data.quantile(0.25):.2f}")
        
        return
    
    def categorical_features(self, column_name):
        """
        Analyze the distribution of values in the specified activity column.

        Parameters
        ----------
        activity_column : str
            Name of the activity column.

        Returns
        -------
        dict
            Dictionary containing counts and percentages of 1s and 0s.
            
        Example usage:
            analyzer = TargetAnalizer(df)

        # description of selected column
            summary = analyzer.categorical_features(column_name)
        """
        counts = self.dataframe[column_name].value_counts()
        percentages = counts / len(self.dataframe) * 100
        
        # Plot pie chart
        colors = ['orange', 'red']
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Distribuzione delle attività')
        plt.show()
        
        # Print results
        print("Numero di 1 nella colonna 'activity':", counts[1])
        print("Numero di 0 nella colonna 'activity':", counts[0])
        print("Percentuale di 1 rispetto al totale:", percentages[1], "%")
        print("Percentuale di 0 rispetto al totale:", percentages[0], "%")
        
        return {'counts': counts, 'percentages': percentages}
    
class Molecular:
    def __init__(self):
        pass
    
    def load_molecules(self, smiles_list):
        """
        Create RDKit Molecule objects from a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings.

        Returns
        -------
        list of RDKit Molecule objects, list of str
            List of successfully loaded molecules and list of problematic SMILES strings.
        
        Example usage:
        loader = Molecular()
        molecules, problematic = loader.load_molecules([smiles_list_str])
        print("Loaded molecules:", molecules)
        print("Problematic SMILES:", problematic)
        
        """

        molecules = []
        problematic_smiles = []

        for smiles in smiles_list:
            try:
                molecule = Chem.MolFromSmiles(smiles, sanitize=False)
                molecules.append(molecule)
            except:
                problematic_smiles.append(smiles)

        print(f"{len(molecules)} molecules were loaded")
        print(f"{len(problematic_smiles)} molecules get error")

        return molecules, problematic_smiles
    
    def set_dative_bonds(self, mol, fromAtoms=(7,8)):
        """ convert some bonds to dative

        Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
        with dative bonds. The replacement is only done if the atom has "too many" bonds.

        Returns the modified molecule.
        
        Parameters
        ----------
        mol : list of RDKit Molecule objects, list of str
            List of mol object from RDKit package.

        Returns
        -------
        list of RDKit Molecule objects, list of str
            List of successfully converted molecules with dative bonds.


        Example usage:
        molecules_list = [loader.set_dative_bonds(m) for m in molecules]
        print(len(molecules_list),"molecules were converted")
        
        """
        def is_transition_metal(at):
            n = at.GetAtomicNum()
            return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

        pt = Chem.GetPeriodicTable()
        rwmol = Chem.RWMol(mol)
        rwmol.UpdatePropertyCache(strict=False)
        metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
        for metal in metals:
            for nbr in metal.GetNeighbors():
                if nbr.GetAtomicNum() in fromAtoms and \
                   nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                   rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                    rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                    rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
        return rwmol
    
    def generate_ECFP(self, molecules_list, max_size = 4, n_bits = 1024):
        """
        ECFP generator
        Parameters
        ----------
        molecules_list : list of RDKit Molecule objects, list of str
            List of mol object from RDKit package.

        max_size : int
            maximum diameter for ECFP

        n_bits : int
            ECFP bit lenght

        Returns
        -------
        ECFP pandas dataframe

        Example usage:

        """
        fp_list = []

        for mol in molecules_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, max_size, nBits=n_bits, useFeatures=True)
            fp_dict = {f'ECFP{i+1}': bit for i, bit in enumerate(fp.ToBitString())}
            fp_list.append(fp_dict)

        ECFP = pd.DataFrame(fp_list)

        return ECFP
    
    def load_Alvadesc(self, descr = "alvadesc", target_col = None, scaler = True):
        """
        load and format alvadesc descriptors.

        Parameters
        ----------
        descriptor : str
            Descriptor class name.
        target_col : str, optional
            Name of the target columns for supervised learning
        scaler : bool
        if true, robust scaler is applied on features

        Returns
        -------
        descriptor : pd.Dataframe
        Descriptor loaded.
        
        Example usage:
        descr = molecular.load_Alvadesc(descriptor = "CATS2D", target_col = df["activity"])        
        """
        
        descriptor = pd.read_csv(f"{descr}.txt",delimiter="\t")
        descriptor = descriptor.drop(columns=["No.","NAME"])
        
        if scaler is True:
            scaler = RobustScaler()
            descriptor_scaled = descriptor.copy()
            descriptor_scaled = scaler.fit_transform(descriptor_scaled)
            
            if target_col is None:
                return descriptor
            else:
                descriptor = pd.concat([descriptor,target_col],axis=1)
                return descriptor
        else:
            if target_col is None:
                return descriptor
            else:
                descriptor = pd.concat([descriptor,target_col],axis=1)
                return descriptor
            
class PrincipalComponentAnalysis:
    def __init__(self):
        pass
    
    def plot2D(self, data, hue_data=None, pc_x='PC1', pc_y='PC2', palette='viridis', figsize=(5,5), dot = 4):
        """
        Perform PCA on data and create a scatter plot of the results.

        Parameters
        ----------
        data : pd.DataFrame
            Data to perform PCA on.
        hue_data : pd.Series, optional
            Data to use for the hue of the scatter plot points. Default is None.
        pc_x : str, optional
            Principal component to plot on the x-axis. Default is 'PC1'.
        pc_y : str, optional
            Principal component to plot on the y-axis. Default is 'PC2'.
        palette : str, optional
            Palette to use for the scatter plot points. Default is 'viridis'.
        figsize : tuple, optional
            Size of the figure. Default is (5,5).
        dot : int
            Size of dots. Default is 4

        Returns
        -------
        df_pca : pd.Dataframe
        PC1/PC2 results dataframe.
        
        Example usage:
        pca_plot2D(data, hue_data, pc_x='PC1', pc_y='PC2', palette='viridis', figsize=(5,5), dot = 4)
        """
        pca = PCA(n_components=2)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data)
        
        pca_result = pca.fit_transform(features_scaled)

        df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=data.index)

        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=figsize)
        
        if hue_data is None:
            sns.scatterplot(data=df_pca, x=pc_x, y=pc_y, s=dot)
        else:
            sns.scatterplot(data=df_pca, x=pc_x, y=pc_y, hue=hue_data, palette=palette, s=dot)

        plt.xlabel('Principal Component {} ({:.2f}%)'.format(pc_x[2:], explained_variance[int(pc_x[2:]) - 1] * 100))
        plt.ylabel('Principal Component {} ({:.2f}%)'.format(pc_y[2:], explained_variance[int(pc_y[2:]) - 1] * 100))
        plt.title('Principal Component Analysis')

        plt.show()

        return df_pca
    
    def chemical_space_selection(self, pca_df, target = 'index', n_select = 1000, figsize=(5,5), dot = 4):
        """
        Parameters
        ----------
        pca_df : pd.DataFrame
            results from PCA.plot2D with index column dropped.
        target : str
            index column name to drop.
        n_select : int
            number of features to select.
        figsize : tuple, optional
            Size of the figure. Default is (5,5).
        dot : int
            Size of dots. Default is 4

        Returns
        -------
        selected_samples : pd.Dataframe
        selected samples with index column.        
        
        Example usage:
        selected_samples = pca.chemical_space_selection(pca_df_inactive, target = 'index', n_select = 1000, figsize=(5,5), dot = 4)
        """
        data = pca_df.copy()
        
        pca_df = pca_df.drop(columns = [target])
        
        weights = np.ones(len(pca_df)) / len(pca_df)
        
        np.random.seed(42)
        
        selected_indices = np.random.choice(len(pca_df), size=n_select, replace=False, p=weights)

        selected_samples = data.iloc[selected_indices, :]

        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], label='all inactive samples', alpha=0.5, s=dot)
        plt.scatter(selected_samples.iloc[:, 0], selected_samples.iloc[:, 1], label='selected inactive samples', alpha=0.5)
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Chemical Space sovrapposition')
        plt.show()
        
        return selected_samples
    
class FeaturesEngineering():
    def __init__(self):
        pass
    
    def preprocess_split_classification(self, dataset, target_col='activity', test_size=0.2, corr_threshold=0.90):
        """
        Parameters
        ----------
        dataset : pd.DataFrame
            dataset to split.
        target_col : str
            name of target column.
        test_size : float
            train/test split ratio. Default is 0.2
        corr_threshold : float
            feature correlation threashold. Default is 0.90

        Returns
        -------
        X, y, X_val, y_val : pd.Dataframe, Series
        train/validation set with respective target.        
        
        Example usage:
        X, y, X_val, y_val = preprocess_and_print_info(dataset, target_col='activity', test_size=0.2, corr_threshold=0.90)        
        """
        print("Initial shape: ", dataset.shape)
        print("#############################")

        print("train-test/val split data:")
        dtf_train, dtf_val = split_dataset(dataset, target_col=target_col, test_size=test_size)

        print("Train set: ", dtf_train.shape)
        print("Validation set: ", dtf_val.shape)
        print("#############################")
        print("Dimension after data preprocessing:")
        X = dtf_train.drop(columns=[target_col])
        y = dtf_train[target_col]
        X = remove_duplicate_columns(remove_semi_constant_columns(remove_constant_columns(X)))
        X = remove_highly_correlated_columns(X, threshold=corr_threshold)
        X_preproc = X

        X_val = dtf_val.drop(columns=[target_col])
        y_val = dtf_val[target_col]

        X_val = X_val[X_preproc.columns]
        print(X.shape, X_val.shape)

        X_copy = X
        X_val_copy = X_val

        dtf_train_new = pd.concat([X, y], axis=1)
        counts = dtf_train_new[target_col].value_counts()

        # Calcolo delle percentuali rispetto al totale
        percentages = counts / len(dtf_train_new) * 100
        # Stampa dei risultati
        print("Percent of positive class:", percentages[1].round(1), "%")

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        for col in X_val.columns:
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce')

        columns_to_replace = {'[': '(', ']': ')', '<': '_', '>': '_'}
        X.rename(columns=lambda x: x.translate(str.maketrans(columns_to_replace)), inplace=True)
        X_val.rename(columns=lambda x: x.translate(str.maketrans(columns_to_replace)), inplace=True)

        return X, y, X_val, y_val
    
    def features_importance(self, model, X, y, kfold_splits=10):
        
        accuracy_scores = []
        sensitivity_scores = []
        specificity_scores = []
        precision_scores = []
        f1_scores = []

        kfold = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=42)

        feature_importance = np.zeros(len(X.columns))  # Initialize an array to store feature importances

        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            sensitivity_scores.append(recall_score(y_test, y_pred, zero_division=1))
            specificity_scores.append(recall_score(y_test, y_pred, pos_label=0, zero_division=1))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=1))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=1))

            feature_importance += model.feature_importances_

        accuracy = np.mean(accuracy_scores)
        sensitivity = np.mean(sensitivity_scores)
        specificity = np.mean(specificity_scores)
        precision = np.mean(precision_scores)
        f1_measure = np.mean(f1_scores)

        feature_importance /= kfold_splits

        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        return feature_importance_df

    def rfe(self, model, X, y, feature_importance_df, mean=False):    
        
        f1_scores_train_k5 = []
        f1_scores_train_k10 = []
        f1_scores_train_k15 = []

        accuracy_scores = []
        sensitivity_scores = []
        specificity_scores = []
        precision_scores = []

        kfold_splits_list = [5, 10, 15]

        for kfold_splits in kfold_splits_list:
            kfold = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=42)

            for i in tqdm(range(X.shape[1], 0, -1), desc=f"k = {kfold_splits}"):
                X_subset = X.iloc[:, :i]

                for train_index, test_index in kfold.split(X_subset, y):
                    X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy_scores.append(accuracy_score(y_test, y_pred))
                    sensitivity_scores.append(recall_score(y_test, y_pred, zero_division=1))
                    specificity_scores.append(recall_score(y_test, y_pred, pos_label=0, zero_division=1))
                    precision_scores.append(precision_score(y_test, y_pred, zero_division=1))
                    f1_scores_train = f1_score(y_test, y_pred, zero_division=1)
                    
                if kfold_splits == 5:
                    f1_scores_train_k5.append(f1_scores_train)
                elif kfold_splits == 10:
                    f1_scores_train_k10.append(f1_scores_train)
                elif kfold_splits == 15:
                    f1_scores_train_k15.append(f1_scores_train)

            accuracy = np.mean(accuracy_scores)
            sensitivity = np.mean(sensitivity_scores)
            specificity = np.mean(specificity_scores)
            precision = np.mean(precision_scores)
         
        f1_scores_train_k5.reverse()
        f1_scores_train_k10.reverse()
        f1_scores_train_k15.reverse()
        f1_scores_train_k5 = pd.Series(f1_scores_train_k5)
        f1_scores_train_k10 = pd.Series(f1_scores_train_k10)
        f1_scores_train_k15 = pd.Series(f1_scores_train_k15)
            
        rolling_mean_k5 = f1_scores_train_k5.rolling(window=10, min_periods=1).mean()
        rolling_mean_k10 = f1_scores_train_k10.rolling(window=10, min_periods=1).mean()
        rolling_mean_k15 = f1_scores_train_k15.rolling(window=10, min_periods=1).mean()
            
        x = feature_importance_df['Feature']
            
        fig, ax1 = plt.subplots(figsize=(20, 6))
                        
        if mean == True:
            ax1.plot(x, f1_scores_train_k5, label='f1_scores_train_k5', color='purple')
            ax1.plot(x, f1_scores_train_k10, label='f1_scores_train_k10', color='green')
            ax1.plot(x, f1_scores_train_k15, label='f1_scores_train_k15', color='orange')

            ax1.plot(x, rolling_mean_k5, label='Media Mobile f1_scores_train_k5', color='purple', linestyle='--')
            ax1.plot(x, rolling_mean_k10, label='Media Mobile f1_scores_train_k10', color='green', linestyle='--')
            ax1.plot(x, rolling_mean_k15, label='Media Mobile f1_scores_train_k15', color='orange', linestyle='--')

            ax1.set_ylabel('F1')
            ax1.legend(loc='lower right')

            ax2 = ax1.twinx()

            ax2.bar(x, feature_importance_df['Importance'], alpha=0.5, color='y', label='Features')

            ax2.set_ylabel('Features Importance')

            plt.show()
            
        else:
            ax1.plot(x, f1_scores_train_k5, label='f1_scores_train_k5', color='purple')
            ax1.plot(x, f1_scores_train_k10, label='f1_scores_train_k10', color='green')
            ax1.plot(x, f1_scores_train_k15, label='f1_scores_train_k15', color='orange')

            ax1.set_ylabel('F1')
            ax1.legend(loc='lower right')

            ax2 = ax1.twinx()

            ax2.bar(x, feature_importance_df['Importance'], alpha=0.5, color='y', label='Features')

            ax2.set_ylabel('Features Importance')

            plt.show()
            
            return f1_scores_train_k5, f1_scores_train_k10, f1_scores_train_k15
        
class Benchmark():
    def __init__(self):
        pass
    
    def train_models_classification(self, models, X, y, kfold_splits=10):
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
    
            # Calculate average scores
            avg_scores = {metric: np.mean(scores_list) for metric, scores_list in scores.items()}
    
            # Append results to list
            results_list.append([name, *avg_scores.values()])
    
        # Create results DataFrame
        results_df = pd.DataFrame(results_list, columns=columns)
        return results_df
    
    def validate_models_classification(self, models, X, y, X_val, y_val, kfold_splits=10):
    
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
    
    
