from groq import Groq
from utils.data_processor import get_df_context

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_ai_insights(df, question, api_key=None):
    try:
        context = get_df_context(df)
        prompt = f"""You are DataMind AI, an expert data analyst.
Dataset details:
{context}

User question: "{question}"
Answer clearly with bullet points."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"
