import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any
import time
from datetime import datetime

# Constants
API_URL = "http://localhost:8000"
AVAILABLE_DBS = ["pinecone", "weaviate", "faiss", "chroma", "elasticsearch"]

def generate_sample_data(n_samples: int = 100) -> tuple:
    """Generate sample fraud data for testing."""
    np.random.seed(42)
    vectors = np.random.randn(n_samples, 128).tolist()  # 128-dimensional vectors
    metadata = [
        {
            "id": f"sample_{i}",
            "is_fraud": np.random.choice([0, 1], p=[0.9, 0.1]),
            "amount": np.random.uniform(10, 10000),
            "timestamp": pd.Timestamp.now().isoformat(),
            "merchant": np.random.choice(["Retail", "Online", "Food", "Travel", "Entertainment"]),
            "location": np.random.choice(["US", "UK", "EU", "Asia", "Other"]),
            "features": {
                "time_of_day": np.random.randint(0, 24),
                "day_of_week": np.random.randint(0, 7),
                "amount_category": np.random.choice(["low", "medium", "high"]),
                "merchant_category": np.random.choice(["retail", "online", "food", "travel"])
            }
        }
        for i in range(n_samples)
    ]
    return vectors, metadata

def initialize_db(db_name: str) -> bool:
    """Initialize the selected vector database."""
    try:
        response = requests.post(f"{API_URL}/initialize/{db_name}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error initializing {db_name}: {str(e)}")
        return False

def insert_data(db_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
    """Insert data into the selected vector database."""
    try:
        response = requests.post(
            f"{API_URL}/insert",
            json={
                "db_name": db_name,
                "vectors": vectors,
                "metadata": metadata
            }
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error inserting data into {db_name}: {str(e)}")
        return False

def evaluate_performance(db_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate performance metrics for the selected database."""
    try:
        response = requests.post(
            f"{API_URL}/evaluate",
            json={
                "db_name": db_name,
                "vectors": vectors,
                "metadata": metadata
            }
        )
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        st.error(f"Error evaluating {db_name}: {str(e)}")
        return {}

def plot_performance_metrics(metrics_df: pd.DataFrame):
    """Create performance visualization plots."""
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Overview", "Detailed Metrics", "Time Analysis", "Fraud Detection"])
    
    with tab1:
        # Radar chart for overall performance
        fig_radar = go.Figure()
        for idx, row in metrics_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['avg_precision'], row['avg_recall'], 
                   1/row['avg_search_time'], 1/row['insertion_time']],
                theta=['Precision', 'Recall', 'Search Speed', 'Insert Speed'],
                fill='toself',
                name=row['database']
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Overall Performance Comparison"
        )
        st.plotly_chart(fig_radar)
        
    with tab2:
        # Detailed metrics comparison
        metrics_to_plot = ['avg_precision', 'avg_recall', 'avg_search_time', 'insertion_time']
        fig_metrics = px.bar(
            metrics_df,
            x="database",
            y=metrics_to_plot,
            title="Detailed Performance Metrics",
            barmode="group"
        )
        st.plotly_chart(fig_metrics)
        
    with tab3:
        # Time analysis
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=metrics_df['database'],
            y=metrics_df['avg_search_time'],
            name='Search Time'
        ))
        fig_time.add_trace(go.Bar(
            x=metrics_df['database'],
            y=metrics_df['insertion_time'],
            name='Insertion Time'
        ))
        fig_time.update_layout(
            title="Time Performance Analysis",
            yaxis_title="Time (seconds)",
            barmode='group'
        )
        st.plotly_chart(fig_time)
        
    with tab4:
        # Fraud detection performance
        fig_fraud = go.Figure()
        fig_fraud.add_trace(go.Bar(
            x=metrics_df['database'],
            y=metrics_df['avg_precision'],
            name='Precision'
        ))
        fig_fraud.add_trace(go.Bar(
            x=metrics_df['database'],
            y=metrics_df['avg_recall'],
            name='Recall'
        ))
        fig_fraud.update_layout(
            title="Fraud Detection Performance",
            yaxis_title="Score",
            barmode='group'
        )
        st.plotly_chart(fig_fraud)

def main():
    st.set_page_config(page_title="Fraud RAG Vector DB Evaluation", layout="wide")
    
    st.title("Fraud RAG Vector Database Evaluation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Database selection
        selected_dbs = st.multiselect(
            "Select Vector Databases to Compare",
            AVAILABLE_DBS,
            default=["faiss", "chroma"]
        )
        
        # Remote database configuration
        st.subheader("Remote Database Configuration")
        
        # Elasticsearch configuration
        if "elasticsearch" in selected_dbs:
            st.write("Elasticsearch Configuration")
            es_url = st.text_input("Elasticsearch URL", value="http://localhost:9200")
            es_username = st.text_input("Elasticsearch Username", value="elastic")
            es_password = st.text_input("Elasticsearch Password", type="password")
            
        # Weaviate configuration
        if "weaviate" in selected_dbs:
            st.write("Weaviate Configuration")
            weaviate_url = st.text_input("Weaviate URL", value="http://localhost:8080")
            weaviate_api_key = st.text_input("Weaviate API Key", type="password")
            
        # Data generation settings
        st.subheader("Data Generation")
        n_samples = st.slider("Number of Samples", 10, 1000, 100)
        fraud_rate = st.slider("Fraud Rate (%)", 1, 20, 10) / 100
        
    # Main content
    st.header("Data Generation")
    
    if st.button("Generate Sample Data"):
        vectors, metadata = generate_sample_data(n_samples)
        st.session_state.vectors = vectors
        st.session_state.metadata = metadata
        
        # Display sample data
        df = pd.DataFrame(metadata)
        
        # Data distribution visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution
            fraud_dist = df['is_fraud'].value_counts()
            fig_fraud_dist = px.pie(
                values=fraud_dist.values,
                names=['Legitimate', 'Fraud'],
                title="Fraud Distribution"
            )
            st.plotly_chart(fig_fraud_dist)
            
        with col2:
            # Amount distribution
            fig_amount = px.histogram(
                df,
                x="amount",
                color="is_fraud",
                title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig_amount)
        
        # Initialize and insert data for selected databases
        for db_name in selected_dbs:
            with st.spinner(f"Processing {db_name}..."):
                if initialize_db(db_name):
                    if insert_data(db_name, vectors, metadata):
                        st.success(f"Successfully processed {db_name}")
                    else:
                        st.error(f"Failed to insert data into {db_name}")
                else:
                    st.error(f"Failed to initialize {db_name}")
    
    # Performance Evaluation
    if "vectors" in st.session_state and "metadata" in st.session_state:
        st.header("Performance Evaluation")
        
        metrics_data = []
        for db_name in selected_dbs:
            with st.spinner(f"Evaluating {db_name}..."):
                metrics = evaluate_performance(
                    db_name,
                    st.session_state.vectors,
                    st.session_state.metadata
                )
                if metrics:
                    metrics["database"] = db_name
                    metrics_data.append(metrics)
        
        if metrics_data:
            # Create metrics DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display metrics table
            st.write("Performance Metrics:")
            st.dataframe(metrics_df)
            
            # Plot performance visualizations
            plot_performance_metrics(metrics_df)
            
            # Additional analysis
            st.subheader("Additional Analysis")
            
            # Time series of performance
            if "timestamp" in st.session_state.metadata[0]:
                df = pd.DataFrame(st.session_state.metadata)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig_timeseries = px.scatter(
                    df,
                    x="timestamp",
                    y="amount",
                    color="is_fraud",
                    title="Transaction Timeline"
                )
                st.plotly_chart(fig_timeseries)
            
            # Merchant analysis
            merchant_fraud = pd.DataFrame(st.session_state.metadata).groupby('merchant')['is_fraud'].mean()
            fig_merchant = px.bar(
                merchant_fraud,
                title="Fraud Rate by Merchant"
            )
            st.plotly_chart(fig_merchant)

if __name__ == "__main__":
    main() 