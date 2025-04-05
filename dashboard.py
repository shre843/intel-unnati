import re
import ast
from typing import List, Tuple, Dict
import tokenize
from io import StringIO
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

class CodeAnalyzer:
    def __init__(self):
        self.bug_patterns = {}
        self.bug_fixes = defaultdict(list)
        self.bug_severity = {}
        self.bug_frequency = {}
        self.df = None
        self.vectorizer = TfidfVectorizer()
        self.code_vectors = None

    def load_dataset(self, file):
        try:
            self.df = pd.read_csv(file)
            self.df['Frequency'] = pd.to_numeric(self.df['Frequency'], errors='coerce')

            for _, row in self.df.iterrows():
                buggy_code = str(row['Buggy Code']).strip()
                fixed_code = str(row['Fixed Code']).strip()
                bug_type = str(row['Bug Type'])
                severity = str(row['Severity'])
                frequency = float(row['Frequency'])
                language = str(row['Programming Language'])

                pattern = self.create_pattern_from_code(buggy_code)

                self.bug_patterns[pattern] = {
                    'description': f'{bug_type} bug detected',
                    'fix': fixed_code,
                    'severity': severity,
                    'frequency': frequency,
                    'language': language,
                    'original_code': buggy_code
                }

                self.bug_fixes[bug_type].append({
                    'pattern': pattern,
                    'fix': fixed_code
                })

                self.bug_severity[bug_type] = severity
                self.bug_frequency[bug_type] = frequency

            buggy_codes = self.df['Buggy Code'].astype(str).values
            self.code_vectors = self.vectorizer.fit_transform(buggy_codes)

            print(f"Successfully loaded dataset with {len(self.bug_patterns)} patterns")

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

    def create_pattern_from_code(self, code: str) -> str:
        pattern = re.escape(code)
        pattern = re.sub(r'[a-zA-Z_]\w*', r'[a-zA-Z_]\\w*', pattern)
        return pattern

    def predict_total_bugs(self):
        if self.df is not None:
            total = len(self.df)
            by_type = self.df.groupby('Bug Type').agg({
                'Fixed Code': lambda x: list(x.unique()),
                'Severity': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
                'Bug Type': 'count'
            }).rename(columns={'Bug Type': 'Count'}).reset_index()

            return {
                'total_bugs': total,
                'bug_details': by_type
            }
        else:
            return {
                'total_bugs': 0,
                'bug_details': pd.DataFrame()
            }

# Streamlit App Title
st.set_page_config(page_title="Code Bug Analyzer Dashboard", layout="wide")
st.title("🐞 Code Bug Analyzer Dashboard")
st.write("Analyze buggy code snippets and suggest fixes using similarity and TF-IDF.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your bug dataset (CSV)", type=["csv"], help="Ensure the CSV contains columns: 'Bug Type', 'Severity', 'Frequency', 'Buggy Code', 'Fixed Code'")

if uploaded_file:
    analyzer = CodeAnalyzer()
    analyzer.load_dataset(uploaded_file)
    stats = analyzer.predict_total_bugs()
    df = analyzer.df

    st.success("Dataset successfully loaded!")

    # Display Data Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_columns = {"Bug Type", "Severity", "Frequency"}
    if not required_columns.issubset(df.columns):
        st.error("Dataset missing required columns! Ensure columns: 'Bug Type', 'Severity', 'Frequency'")
    else:
        # Bug Type Frequency Bar Chart
        st.subheader("🔢 Bug Type Frequency")
        bug_counts = df["Bug Type"].value_counts()
        fig, ax = plt.subplots()
        bug_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_xlabel("Bug Type")
        ax.set_ylabel("Count")
        ax.set_title("Bug Type Distribution")
        st.pyplot(fig)
        
        # Bug Severity Pie Chart
        st.subheader("🚦 Bug Severity Distribution")
        severity_counts = df["Severity"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', colors=["red", "orange", "yellow", "green"], startangle=140)
        ax.set_title("Bug Severity Breakdown")
        st.pyplot(fig)
        
        # Bug Frequency Heatmap (Optional)
        st.subheader("🔥 Bug Frequency Heatmap")
        plt.figure(figsize=(10, 5))
        heatmap_data = df.pivot_table(index='Bug Type', values='Frequency', aggfunc='sum')
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True)
        st.pyplot(plt)

