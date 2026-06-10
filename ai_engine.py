from groq import Groq
from utils.data_processor import get_df_context

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_ai_insights(df, question, api_key=None):
    pass 
