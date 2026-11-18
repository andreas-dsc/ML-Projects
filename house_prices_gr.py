#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# In[2]:


# ==========================================
# 1. DEFINE CLEANING FUNCTIONS FOR PIPELINE
# ==========================================
def clean_greek_num_series(series):
    return pd.to_numeric(
        series.astype(str)
              .str.strip()
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def raw_data_cleaning_pipeline(df_input):
    """
    Pipeline step to clean raw real estate data.
    """
    df = df_input.copy()
    
    # Drop unnecessary columns if they exist in raw data
    cols_to_drop = [c for c in ['Source.Name', 'Νομαρχία', 'Δήμος Καλλικράτη'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    # Map APAA indicator
    if "Ένδειξη ΑΠΑΑ" in df.columns:
        df["Ένδειξη ΑΠΑΑ"] = df["Ένδειξη ΑΠΑΑ"].map({"Εντός ΑΠΑΑ": True, "Εκτός ΑΠΑΑ": False}).astype(int)
        df = df.rename(columns={"Ένδειξη ΑΠΑΑ": "isthere_APAA"})
        
    # Filter for houses or apartments
    if "Κατηγορία Ακινήτου" in df.columns:
        df = df[df["Κατηγορία Ακινήτου"].isin(["Κατοικία ή διαμέρισμα πλήν μονοκατοικίας", "Μονοκατοικία"])]
        
    # Filter for full ownership
    if "Είδος Εμπράγματου δικαιώματος Κτίσματος" in df.columns:
        df = df[df["Είδος Εμπράγματου δικαιώματος Κτίσματος"] == "Πλήρης Κυριότητα"]
        df = df.drop(columns=["Είδος Εμπράγματου δικαιώματος Κτίσματος"])
        
    # Drop special conditions if present
    if "Ειδικές Συνθήκες Ακινήτου" in df.columns:
        df = df[df["Ειδικές Συνθήκες Ακινήτου"].isna()]
        df = df.drop(columns=["Ειδικές Συνθήκες Ακινήτου"])
        
    # Clean numeric columns
    numeric_cols = [
        "Eπιφάνεια Κύριων Χώρων (σε τ.μ.)", 
        "Τίμημα Δικαιώματος", 
        "Επιφάνεια Βοηθητικών Χώρων (σε τ.μ.)",
        "Επιφάνεια Οικοπέδου (σε τ.μ.)",
        "Συνολική Επιφάνεια Κτισμάτων στο οικόπεδο"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_greek_num_series(df[col])
            
    # Fill specific missing defaults
    if "Επιφάνεια Βοηθητικών Χώρων (σε τ.μ.)" in df.columns:
        df["Επιφάνεια Βοηθητικών Χώρων (σε τ.μ.)"] = df["Επιφάνεια Βοηθητικών Χώρων (σε τ.μ.)"].fillna(0.0)
    if "Επιφάνεια Οικοπέδου (σε τ.μ.)" in df.columns:
        df["Επιφάνεια Οικοπέδου (σε τ.μ.)"] = df["Επιφάνεια Οικοπέδου (σε τ.μ.)"].fillna(0.0)
        
    # Drop rows with nulls in crucial constraints
    subset_notna = [c for c in ["Έτος Κατασκευής", "Ποσοστό Συνιδιοκτησίας Κτίσματος"] if c in df.columns]
    if subset_notna:
        df = df.dropna(subset=subset_notna)
        
    # Process dates if present
    if "Ημερομηνία Συμβολαίου" in df.columns:
        dates = pd.to_datetime(df["Ημερομηνία Συμβολαίου"], dayfirst=True, errors="coerce")
        df["year"] = dates.dt.year
        df["month"] = dates.dt.month
        df["decimal_year"] = df["year"] + (dates.dt.dayofyear - 1) / 365.25
        df["building_age"] = (df["year"] - df["Έτος Κατασκευής"]).clip(lower=0)
        df = df.drop(columns=["Ημερομηνία Συμβολαίου", "year", "month"])
        
    # Drop redundant sparse land ownership columns if they exist
    land_cols = [
        "Ποσοστό Συνιδιοκτησίας Οικοπέδου", 
        "Είδος Εμπράγματου δικαιώματος Οικοπέδου", 
        "Συνολική Επιφάνεια Κτισμάτων στο οικόπεδο"
    ]
    df = df.drop(columns=[c for c in land_cols if c in df.columns])
    
    # Handle floor ("Όροφος") basement mapping
    if "Όροφος" in df.columns:
        df["Όροφος"] = df["Όροφος"].replace("Υ", -1).astype(int)
        
    # Convert property category to binary flag
    if "Κατηγορία Ακινήτου" in df.columns:
        df["is_detached_house"] = (df["Κατηγορία Ακινήτου"] == "Μονοκατοικία").astype(int)
        df = df.drop(columns=["Κατηγορία Ακινήτου"])
        
    return df.reset_index(drop=True)

# ==========================================
# 2. RUN PIPELINE & TRAIN MODEL
# ==========================================
# Load raw dataset
df_raw = pd.read_csv("Merged_Excel_files.csv", sep=";")

# Apply cleaning pipeline
df_cleaned = raw_data_cleaning_pipeline(df_raw)

# Filter target outliers
df_cleaned = df_cleaned[df_cleaned['Τίμημα Δικαιώματος'] < 2000000]

# Define Features and Target
X = df_cleaned.drop(columns=['Τίμημα Δικαιώματος'])
y = df_cleaned['Τίμημα Δικαιώματος']

# Calculate target encoding for areas
area_means = y.groupby(X['Δημοτικό ή Κοινοτικό Διαμέρισμα']).mean()
X['Area_Target_Encoded'] = X['Δημοτικό ή Κοινοτικό Διαμέρισμα'].map(area_means)
X = X.drop(columns=['Δημοτικό ή Κοινοτικό Διαμέρισμα'])

# Train Model to the whole dataset, the R2 score is close to 0.65 (not shown here)
gb = HistGradientBoostingRegressor(random_state=42)
gb.fit(X, y)

# ==========================================
# 3. PREDICT FOR THE NEW CUSTOMER
# ==========================================
area = 'ΑΘΗΝΑΙΩΝ'
size_sqm = 90.0
build_year = 1999.0
floor = 2
current_year = 2026

athens_historical = df_cleaned[df_cleaned['Δημοτικό ή Κοινοτικό Διαμέρισμα'] == area]
estimated_zone_price = athens_historical['Tιμή Ζώνης'].median()
typical_facades = athens_historical['Πλήθος Προσόψεων'].mode()[0]

new_listing = pd.DataFrame([{
    'isthere_APAA': 1,
    'Πλήθος Προσόψεων': typical_facades,
    'Tιμή Ζώνης': estimated_zone_price,
    'Eπιφάνεια Κύριων Χώρων (σε τ.μ.)': size_sqm,
    'Επιφάνεια Βοηθητικών Χώρων (σε τ.μ.)': 0.0,
    'Έτος Κατασκευής': build_year,
    'Ποσοστό Συνιδιοκτησίας Κτίσματος': 100.0,
    'Όροφος': floor,
    'Επιφάνεια Οικοπέδου (σε τ.μ.)': 0.0,
    'decimal_year': 2026.78,
    'building_age': current_year - build_year,
    'is_detached_house': 0
}])

new_listing['Area_Target_Encoded'] = area_means[area]
new_listing = new_listing[X.columns]

predicted_price = gb.predict(new_listing)[0]
print(f"Estimated Market Value: €{predicted_price:,.0f}")


# In[ ]:




