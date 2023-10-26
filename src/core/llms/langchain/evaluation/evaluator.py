import argparse
import os
from typing import Any, Dict, List, Tuple

from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.base import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake

logger = get_logger()


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.embedding_model = OpenAIEmbeddings()
        self.vectorstore = self.load_vectorstore()
        self.qa_with_sources_chain = load_qa_with_sources_chain(
            llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0),
            prompt=QA_WITH_SOURCES_PROMPT,
        )

        self.model_runner = OpenAiModelRunner(
            api_key=os.environ["OPENAI_API_KEY"],
            project_name="llm-question_answering",
            in_token_limit=2000,
        )

    def load_vectorstore(self) -> DeepLake:
        model_dir = "deploy/model/multisource"
        vectorstore = DeepLake(
            dataset_path=model_dir,
            embedding_function=self.embedding_model,
            read_only=True,
        )
        return vectorstore

    def get_response_from_question(self, question: str) -> Dict[str, Any]:
        """
        Given a question, return dict with
            `question`
            `input_documents`
            `output_text`
        """
        sources = self.vectorstore.similarity_search(question, k=3)
        response = self.qa_with_sources_chain(
            {"question": question, "input_documents": sources}
        )
        return response

    def eval_with_rubric(self, response):
        chain = LLMChain(
            llm=OpenAI(model_name="gpt-4", temperature=0),
            prompt=EVAL_WITH_RUBRIC_PROMPT,
        )
        evaluated_response = chain.run(
            question=response["question"],
            context=response["input_documents"],
            completion=response["output_text"],
        )
        return evaluated_response


# args: user, output_name,
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="multisource")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    questions = [
        "How can a customer can change the Representative's Contact Information for their texting compliance settings?",
        "How do I customize a rentable item report to only show available units?",
        "what does a rent roll report do?",
        "How do I refund security deposits from escrow?",
        "why are our vendors not receiving receipt notifications",
        "how do i delete a journal entry",
        "Can you change the automated email response that goes to Guest Cards?",
        "Can you add rentable items to an association. ",
        "Can we prepare owner distribution from prepaid rents?",
        "How would Vendor resume an archived text message chat with a Tenant to finish scheduling a follow up appointment for work order.",
    ]
    evaluation_info = []
    for question in questions:
        logger.info(f"Evaluting question: {question}")
        response = evaluator.get_response_from_question(question)
        evaluated_response = evaluator.eval_with_rubric(response)
        ei = {
            "question": question,
            "context": response["input_documents"],
            "completion": response["output_text"],
            "evaluated_response": evaluated_response,
        }
        evaluation_info.append(ei)
    print(evaluation_info)
