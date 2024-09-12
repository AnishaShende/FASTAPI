# fastapi_app.py
import time
import json
import csv
import faiss
import uvicorn
import logging
import threading
import numpy as np
from PyPDF2 import PdfReader
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from lightrag.core.component import Component
from lightrag.core.generator import Generator
from sentence_transformers import SentenceTransformer
from lightrag.components.model_client import OllamaClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)

# FastAPI app initialization
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual Vercel app URL
    # allow_origins=["https://nyayalay.vercel.app/", "https://nyayalay-gm2xkbvrc-parthnarkhedes-projects.vercel.app/" ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings_path = "data/embeddings.npy"
chunks_path = "data/chunks.json"

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed embeddings and chunks
def load_precomputed_data(embeddings_path, chunks_path):
    embeddings = np.load(embeddings_path)
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    return embeddings, chunks

# Load the precomputed embeddings and chunks
embeddings, chunks = load_precomputed_data(embeddings_path, chunks_path)

# Vector store class
class VectorStore:
    def __init__(self, embeddings, chunks, embedder):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks
        self.embedder = embedder

    def search(self, query, k=5):
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        return [self.chunks[i] for i in indices[0]]

# Define the request body model for FastAPI
class QueryRequest(BaseModel):
    question: str

# Initialize the vector store
vector_store = VectorStore(embeddings, chunks, embedder)

# Function to log detailed context information for debugging
def log_context_info(context, query):
    logging.info(f"Query: {query}")
    logging.info(f"Context: {context}")

class ContextAwareQA(Component):
    def __init__(self, generator, vector_store):
        super().__init__()
        self.generator = generator
        self.vector_store = vector_store

    def extract_field(self, raw_output, start_token, end_token=None):
      try:
          start_index = raw_output.lower().find(start_token.lower())  # Case insensitive search
          if start_index == -1:
              logging.warning(f"Start token '{start_token}' not found in the output.")
              return None
          start_index += len(start_token)
          if end_token:
              end_index = raw_output.lower().find(end_token.lower(), start_index)
              if end_index == -1:
                  logging.warning(f"End token '{end_token}' not found in the output after '{start_token}'.")
                  return raw_output[start_index:].strip()
              return raw_output[start_index:end_index].strip()
          return raw_output[start_index:].strip()
      except Exception as e:
            logging.error(f"Error extracting field with start_token '{start_token}': {str(e)}")
            return None


    def process_output(self, raw_output):
      json_output = {
          "Data": {
              "Predictive_analysis": "",
              "Similar_cases": []
          }
      }

      # Extracting the Predictive Analysis
      json_output["Data"]["Predictive_analysis"] = self.extract_field(raw_output, "1. Predictive Analysis:", "2. Similar Cases:")

      # Extracting the Similar Cases
      case_name = self.extract_field(raw_output, "Case Name:", "Date:")
      case_date = self.extract_field(raw_output, "Date:", "Decision:")
      case_decision = self.extract_field(raw_output, "Decision:", "Case No:")

      # Extracting Case Details
      case_details = {
          "Case No": self.extract_field(raw_output, "Case No:", "Court:"),
          "Court": self.extract_field(raw_output, "Court:", "Case Status:"),
          "Case Status": self.extract_field(raw_output, "Case Status:", "Judge:"),
          "Judge": self.extract_field(raw_output, "Judge:", "Section:"),
          "Section": self.extract_field(raw_output, "Section:", "Facts:"),
          "Facts": self.extract_field(raw_output, "Facts:", "Legal Issues:"),
          "Legal Issues": self.extract_field(raw_output, "Legal Issues:", "Key Legal Questions:"),
          "Key Legal Questions": self.extract_field(raw_output, "Key Legal Questions:", "Plaintiff Arguments:"),
          "Plaintiff Arguments": self.extract_field(raw_output, "Plaintiff Arguments:", "Defendant Arguments:"),
          "Defendant Arguments": self.extract_field(raw_output, "Defendant Arguments:", "Court's Reasoning:"),
          "Court's Reasoning": self.extract_field(raw_output, "Court's Reasoning:", "Decision:"),
          "Decision": self.extract_field(raw_output, "Decision:", "Conclusion:"),
          "Conclusion": self.extract_field(raw_output, "Conclusion:", "Case Summary:"),
          "Case Summary": self.extract_field(raw_output, "Case Summary:")
      }

      # Creating a case entry
      case_entry = {
          "Case_name": case_name,
          "Date": case_date,
          "Decision": case_decision,
          "case_details": case_details
      }

      # Adding the case entry to the list of similar cases
      json_output["Data"]["Similar_cases"].append(case_entry)

      return json_output


    def call(self, input: dict) -> dict:
        try:
            query = input['input_str']
            context = "\n".join(self.vector_store.search(query))
            response = self.generator.call({"input_str": query, "context": context})

            # Log the raw response for debugging
            logging.info(f"Raw response from the generator: {response.data}")

            # Ensure the response contains data
            if hasattr(response, 'data') and response.data:
                try:
                    # Process the raw output using the process_output method
                    processed_output = self.process_output(response.data)

                    # Return the structured JSON
                    return processed_output

                except json.JSONDecodeError:
                    logging.error("Failed to parse JSON from generator output.")
                    raise HTTPException(status_code=500, detail="Failed to parse JSON from generator output.")
            else:
                logging.error("Invalid response structure from generator.")
                raise HTTPException(status_code=500, detail="Invalid response structure from the QA pipeline.")

        except KeyError as e:
            logging.error(f"KeyError: Missing key in input: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Missing key in input: {str(e)}")
        except Exception as e:
            logging.error(f"Error in ContextAwareQA: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in QA processing: {str(e)}")

# Initialize your QA system
qa_template = r"""
<SYS>
You are an assistant providing precise legal analysis and case summaries for Indian commercial court judges. Generate content that directly fills the required fields without any extra formatting or headings. Ensure that each response is concise, clear, and strictly adheres to the specific field it is meant for.
</SYS>

Context:
{context}

1. Predictive Analysis:
{{predictive_summary}} [Generate a concise summary of the predictive analysis relevant to the case, without any extra text.]

2. Similar Cases:
Case Name: {{case_name}} [Provide only the case name.]
Date: {{case_date}} [Provide only the date of the case.]
Decision: {{case_decision}} [Provide a summary of the decision.]

Case Details:
Case No: {{case_number}} [Provide only the case number.]
Court: {{case_court}} [Provide only the court name.]
Case Status: {{case_status}} [Provide only the case status.]
Judge: {{case_judge}} [Provide only the judge's name.]
Section: {{case_section}} [Provide only the relevant section.]
Facts: {{case_facts}} [Provide a brief summary of the case facts.]
Legal Issues: {{case_legal_issues}} [List the key legal issues.]
Key Legal Questions: {{case_key_questions}} [List the key legal questions.]
Plaintiff Arguments: {{plaintiff_arguments}} [Summarize the plaintiff's arguments.]
Defendant Arguments: {{defendant_arguments}} [Summarize the defendant's arguments.]
Court's Reasoning: {{court_reasoning}} [Summarize the court's reasoning.]
Decision: {{court_decision}} [Provide the final decision.]
Conclusion: {{court_conclusion}} [Summarize the conclusion of the case.]
Case Summary: {{case_summary}} [Provide a concise summary of the case.]

User Query: {user_query}
You:
"""



generator = Generator(
    model_client=OllamaClient(),
    model_kwargs={"model": "llama3.1"},
    template=qa_template,
)

qa_pipeline = ContextAwareQA(generator=generator, vector_store=vector_store)

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API endpoint 7'}


@app.post("/ask")
async def ask_question(query: QueryRequest):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Call the QA pipeline and get the response
            response = qa_pipeline.call({"input_str": query.question})

            # Ensure the response is correctly formatted and contains required fields
            if isinstance(response, dict) and 'Data' in response:
                return response
            else:
                logging.error("Invalid response structure from generator.")
                raise HTTPException(status_code=500, detail="Invalid response structure from the QA pipeline.")

        except ConnectionError:
            if attempt < max_retries - 1:
                logging.warning(f"Connection error. Retrying attempt {attempt + 1}/{max_retries}...")
                time.sleep(2)  # Wait before retrying
            else:
                logging.error("Max retries reached. Connection error.")
                raise HTTPException(status_code=500, detail="Unable to reach the external service after multiple attempts.")
        except HTTPException as e:
            # Reraise HTTPExceptions with more specific messages
            logging.error(f"HTTP error: {e.detail}")
            raise
        except Exception as e:
            # Catch any other unexpected exceptions
            logging.error(f"Unexpected error in attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
