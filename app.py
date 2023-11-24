import yaml

import gradio as gr

from typing import List

from llm_client import PalmClient
from pinecone_index import PinceconeIndex

SYSTEM_MESSAGE = "Give a precise answer to the question based on only the \
            context and evidence and do not be verbose."
TOP_K = 2


def format_prompt(question: str, evidence: List[str]):
    evidence_string = ""
    for i, ev in enumerate(evidence):
        evidence_string.join(f"\n Evidence {i+1}: {ev}")

    content = f"{SYSTEM_MESSAGE} \
              \n ### Question:{question} \
              \n ### Evidence: {evidence_string} \
              \n ### Response:"

    return content


if __name__ == "__main__":
    config_path = "config.yml"
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config["paths"]["data_path"]
    project = config["paths"]["project"]

    index_name = config["pinecone"]["index-name"]
    embedding_model = config["sentence-transformers"]["model-name"]
    embedding_dimension = config["sentence-transformers"]["embedding-dimension"]

    index = PinceconeIndex(index_name, embedding_model)
    index.connect_index(embedding_dimension, False)

    palm_client = PalmClient()

    def get_answer(question: str):
        evidence = index.query(question, top_k=TOP_K)
        prompt_with_evidence = format_prompt(question, evidence)
        print(prompt_with_evidence)
        response = palm_client.generate_text(prompt_with_evidence)
        final_output = [response] + evidence

        return final_output

    context_outputs = [gr.Textbox(label=f"Evidence {i+1}") for i in range(TOP_K)]
    result_output = [gr.Textbox(label="Answer")]

    gradio_outputs = result_output + context_outputs
    gradio_inputs = gr.Textbox(placeholder="Enter your question...")

    demo = gr.Interface(
        fn=get_answer,
        inputs=gradio_inputs,
        # outputs=[gr.Textbox(label=f'Document {i+1}') for i in range(TOP_K)],
        outputs=gradio_outputs,
        title="GT Student Code of Conduct Bot",
        description="Get LLM-powered answers to questions about the \
            Georgia Tech Student Code of Conduct. The evidences are exerpts\
                from the Code of Conduct.",
    )

    demo.launch()
