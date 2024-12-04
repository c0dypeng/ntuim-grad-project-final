
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
   git clone https://github.com/c0dypeng/ntuim-grad-project-final/
   cd course-rag-prototype
   ```
2. Install the required dependencies:
   ```bash
   make env
   ```
3. Set up your environment variables by creating a `.env` file in the project root and adding your Pinecone and TAIDE credentials:
   ```bash
   PINECONE_API_KEY=<your-pinecone-api-key>
   OPENAI_API_KEY=<your-openai-api-key>
   ```
4. Run the Streamlit app:
   ```bash
   make
   ```
   
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
