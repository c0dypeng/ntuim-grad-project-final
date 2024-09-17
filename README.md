
# Course-RAG Prototype

This repository contains the code for a Course Recommendation and Search System utilizing Retrieval-Augmented Generation (RAG) and Pinecone for efficient data retrieval.

## Getting Started

### Prerequisites

Ensure you have the following installed before proceeding:
- Python 3.x
- All dependencies listed in `requirements.txt`

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd course-rag-prototype
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables by creating a `.env` file in the project root and adding your Pinecone and TAIDE credentials:
   ```bash
   PINECONE_API_KEY=<your-pinecone-api-key>
   TAIDE_EMAIL=<your-taide-email>
   TAIDE_PASSWORD=<your-taide-password>
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Known Issues

- **Query Error with `k > 1`**: Setting the `k` value greater than 1 in `streamlit_app.py` may cause an error during query execution. This is currently being investigated.

## Project Structure

- **`data-processing.ipynb`**: Preprocesses the raw data and exports it as `course-v1.csv`.
- **`taide_chat.py`**: Handles connections to the TAIDE large language model (LLM).
- **`upload-pinecone.ipynb`**: Uploads the processed `course-v1.csv` data to Pinecone for vectorized retrieval.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
