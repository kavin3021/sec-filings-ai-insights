import os
import json
import streamlit as st
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer

# Load GCP creds from Streamlit secrets
creds = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
with open("/tmp/key.json", "w") as f:
    json.dump(creds, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/key.json"

st.set_page_config(page_title="SEC Filings AI Insights", layout="wide")

st.title("🔍 SEC Filings AI Insights")
st.write("Enter a natural-language query to explore SEC filings:")

user_query = st.text_input("Search query", placeholder="e.g., technology company risks in 2023")

if user_query:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(user_query)
        embedding_list = ', '.join(f"{v:.18f}" for v in embedding)

        bq = bigquery.Client()
        sql = f"""
        WITH query AS (SELECT [{embedding_list}] AS query_embedding)
        SELECT 
          entityName, fy, form, ml_risk_score, ml_sentiment_classification, 
          risk_summary, financial_summary, ml_top_ngrams, filing_excerpt
        FROM knowledge_base.sec_filings_ml_insights
        ORDER BY ml_risk_score DESC
        LIMIT 10
        """

        df = bq.query(sql).to_dataframe()

        for i, row in df.iterrows():
            with st.expander(f"{row['entityName']} (FY {row['fy']}) – Risk: {row['ml_risk_score']:.1f}"):
                st.markdown(f"**Risk Summary:** {row['risk_summary']}")
                st.markdown(f"**Financial Summary:** {row['financial_summary']}")
                st.markdown(f"**Sentiment:** {row['ml_sentiment_classification']}")
                st.markdown(f"**Top Terms:** {row['ml_top_ngrams']}")
                st.markdown(f"**Excerpt:** {row['filing_excerpt']}")
    except Exception as e:
        st.error(f"Error querying BigQuery: {e}")
