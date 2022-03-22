# -*- coding: utf-8 -*-

"""The methods to load ML models and predict the heat capacity using a set of ML features."""

import numpy as np
import sys
import pandas as pd
from .descriptors import cv_features
import joblib
import glob
import copy

FEATURES = cv_features

def predict_Cv_ensemble_structure(ensemble_models: list, FEATURES: list, df_features: pd.DataFrame, structure_name: str) -> list:
    """Predict heat capacity using an ensemble of ML models for one structure.

    :param ensemble_models: ensemble of ML models
    :param FEATURES: features for ML model
    :param df_features: pandas dataframe containing the features
    :param structure_name: the name of structure

    Returns a list containing the gravimetric and molar heat capacity together with the uncertainty of the models
    """
    df_site_structure = df_features.loc[df_features["structure_name"]==structure_name]
    predictions_gravimetric = []
    predictions_molar = []
    for model_idx, model in enumerate(ensemble_models):
        df_site_structure["pCv_300.00_predicted_%i"%model_idx]=model.predict(df_site_structure[FEATURES])
        predicted_mol = np.sum(df_site_structure["pCv_300.00_predicted_%i"%model_idx])/len(df_site_structure)
        predicted_gr = np.sum(df_site_structure["pCv_300.00_predicted_%i"%model_idx])/np.sum(df_site_structure["site AtomicWeight"])
        predictions_molar.append(predicted_mol)
        predictions_gravimetric.append(predicted_gr)
    
    gr_mean = np.mean(predictions_gravimetric)
    gr_std = np.std(predictions_gravimetric)
    mol_mean = np.mean(predictions_molar)
    mol_std = np.std(predictions_molar)
    
    for ix in df_features.loc[df_features["structure_name"]==structure_name].index:
        df_features.loc[ix,"Cv_gravimetric_predicted_mean"]= gr_mean
        df_features.loc[ix,"Cv_gravimetric_predicted_std"]= gr_std
        df_features.loc[ix,"Cv_molar_predicted_mean"] = mol_mean
        df_features.loc[ix,"Cv_molar_predicted_std"] = mol_std
        
        
    return gr_mean, gr_std, mol_mean, mol_std




def predict_Cv_ensemble_dataset(models: list, FEATURES: list, df_features: pd.DataFrame, temperature: float) -> list:
    """Predict heat capacity using an ensemble of ML models for a dataset.

    :param models: ensemble of ML models
    :param FEATURES: features for ML model
    :param df_features: pandas dataframe containing the features
    :param temperature: target temperature 

    Returns a list containing the gravimetric and molar heat capacity together with the uncertainty of the models
    """
    df_site_structure=copy.deepcopy(df_features)
    for model_idx,model in enumerate(models):
        df_site_structure["pCv_{}_predicted_{}".format(temperature,model_idx)]=model.predict(df_site_structure[FEATURES])
    results=[]
    for name in df_site_structure["structure_name"].unique():
        predicted_mol=[]
        predicted_gr=[]
        for model_idx in range(len(models)):
            sites=df_site_structure.loc[df_site_structure["structure_name"]==name]
            predicted_mol.append(np.sum(sites["pCv_{}_predicted_{}".format(temperature,model_idx)])/len(sites))
            predicted_gr.append(np.sum(sites["pCv_{}_predicted_{}".format(temperature,model_idx)])/np.sum(sites["site AtomicWeight"]))
        results.append({
            "name":name,
            "Cv_gravimetric_{}_mean".format(temperature): np.mean(predicted_gr),
            "Cv_gravimetric_{}_std".format(temperature): np.std(predicted_gr),
            "Cv_molar_{}_mean".format(temperature): np.mean(predicted_mol),
            "Cv_molar_{}_std".format(temperature): np.std(predicted_mol),
        })
    return results


def predict_Cv_ensemble_dataset_multitemperatures(path_to_models: str, features_file: str="features.csv", FEATURES: list=cv_features, temperatures: list=[300.00], save_to: str="cv_predicted.csv") -> pd.DataFrame:
    """Predict heat capacity for multiple temperatures using an ensemble of ML models for a dataset.

    :param path_to_models: directory storing the ML models
    :param FEATURES: features for ML model
    :param df_features: pandas dataframe containing the features
    :param temperature: target temperature 

    Returns a list containing the gravimetric and molar heat capacity together with the uncertainty of the models
    """
    df_features = pd.read_csv(features_file)
    df_features["structure_name"]=["_".join(n.split("_")[:-1]) for n in df_features["Unnamed: 0"]]
    print("predicting Cp for {} structures".format(len(df_features["structure_name"].unique())))
    for i,temperature in enumerate(temperatures):
        models=[]
        print("loading models for:", temperature)
        modelnames = glob.glob("{}/{:.2f}/*".format(path_to_models, temperature))
        models = [joblib.load(n) for n in modelnames]
        print("{} models loaded, predicting...".format(len(models)))
        if i==0:
            res= pd.DataFrame(predict_Cv_ensemble_dataset(models, FEATURES, df_features, temperature))
            all_results=res
        else:
            res= pd.DataFrame(predict_Cv_ensemble_dataset(models, FEATURES,df_features, temperature))
            all_results=all_results.merge(res, how="inner",on="name")

    if save_to:
        all_results.to_csv(save_to)
    return all_results
