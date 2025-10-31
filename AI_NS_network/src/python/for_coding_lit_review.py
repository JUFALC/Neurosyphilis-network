# FoR coding for lit review (clean)
# Project: AI-Powered Classification & Network Mapping
# Version: 1.0 (2025-10-27)
# Notes: CLI-ready, uses project-relative paths and .env for secrets

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise RuntimeError('Set OPENAI_API_KEY in your environment or .env file')
client = OpenAI(api_key=api_key)

# Parameters for LLM
model = "gpt-4o-mini"  # Or "gpt-4" or gpt-3.5-turbo
max_tokens = 100  # Adjust based on response length
temperature = 0.2  # Low temperature to reduce verbosity
top_p = 1  # Keep deterministic sampling

# Read the Excel file that contains texts and elements for evaluation
for_file_path = os.path.join("data", "for_structure_cleaned.xlsx")
df_FoR = pd.read_excel(for_file_path, sheet_name="divisions")

text_file_path = os.path.join("data", "FINAL_data_for_TOPICeval_PART9_missings.xlsx")
df_text = pd.read_excel(text_file_path)


# Function to create prompt with text, element, and additional instructions
def create_prompt(FoR, defin, excl, text):
    # Example of additional instructions (modify based on your theoretical framework)
    instructions = f"""
    The field of research '{FoR}' is defined as: '{defin}' with the following exclusions: '{excl}'.
    Based on this definition, is the following text '{text}' part of the field of research '{FoR}'? 
    Output either 0 or 1, followed by a pipe symbol (|), a brief justification.
    0 means the text is not in the field of research; 1 means it is.
    Be concise.
    """
    return instructions


# Function to send prompt to OpenAI LLM and get the response using the new API
def query_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content  # Get the LLM response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None


# Create a list to store results
results = []

# Nested loops: loop through each text and for each text, loop through each field of research
for text_row in df_text.itertuples():
    text = text_row.Text  # Adjust based on actual column name in the text file
    textid = text_row.id
    for FoR_row in df_FoR.itertuples():
        FoR = FoR_row.Category  # Adjust based on actual column name in the FoR file
        defin = FoR_row.Definition
        excl = FoR_row.Exclusions

        # Construct the prompt for the current text and element combination
        prompt = create_prompt(FoR, defin, excl, text)

        # For debugging or testing, print the constructed prompt
        print(prompt)

        # Query the LLM for the evaluation
        evaluation = query_llm(prompt)

        # Append the result (you can adjust the structure as needed)
        results.append({
            'TextID': textid,
            'Text': text,
            'FoR': FoR,
            'Prompt': prompt,
            'Evaluation': evaluation
        })

# Convert results to DataFrame for further processing or export
results_df = pd.DataFrame(results)

# Save results to a CSV or Excel file for further review
results_df.to_csv("output/llm_evaluation_FOR_results.csv", index=False)
# Or save to Excel
results_df.to_excel("output/llm_evaluation_FOR_results.csv", index=False)
results_df.head()

print("Evaluation complete. Results saved to file.")


if __name__ == "__main__":
    # Ensure project folders exist
    for d in ("data","output","figures"):
        os.makedirs(d, exist_ok=True)
    print("Running FoR coding for lit review ...")
    # (script runs via top-level code)
