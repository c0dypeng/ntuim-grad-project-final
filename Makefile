.PHONY: run env

run:
	streamlit run course-rag-prototype/streamlit_app.py

env:
	pip install -r course-rag-prototype/requirements.txt
