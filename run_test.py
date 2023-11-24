import pandas as pd
from datetime import datetime
import numpy as np


def get_answer(model, query, tokenizer, context, question, max_new_tokens, temperature, 
              num_evidence, evidence=None):
  full_question = context + " " + question
  evidences = query.run_query(full_question, num_evidence)
  all_evidence = ""
  i = 0
  for evidence in evidences:
    all_evidence += str("Evidence " + str(i) + ": " + evidence + " ")
    i += 1
  full_context = f'Give a precise answer to the question based on only the context and evidence and do not be verbose. ### Context:{full_question}\n ### Evidence: {all_evidence}\n ### Answer:'
  inputs = tokenizer(full_context, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, 
  												 return_dict_in_generate=True, output_scores=True)
  input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
  generated_tokens = outputs.sequences[:, input_length:].cpu()
  transition_scores = model.compute_transition_scores(
  outputs.sequences, outputs.scores, normalize_logits=True
  )
  transition_scores = transition_scores.cpu()
  hall = 0
  tot = 0
  output = ""
  for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    output += tokenizer.decode(tok)
    prob = np.exp(score.numpy())
    tot += 1
    if prob <= 0.50:
      hall += 1
  print(hall / tot)
  full_context += "\nHaalucination score: " + str(hall / tot)
  return output, full_context

def get_answer_without_evidence(pipe, question):
	content = f'Answer the following question. {question}'
	outputs = pipe(content, max_new_tokens=256, temperature=0.7)
	return outputs[0]["generated_text"][len(content):]


def run_test_with_evidence(test, model, query, tokenizer):
	questions = pd.read_csv('data/code_of_conduct/questions.csv')
	# make sure to change output file name each time and download locally
	output_file = test[0]
	# output configurations to top of file
	max_new_tokens = test[1]
	temperature = test[2]
	num_evidence = test[3]
	chunking = test[4]

	with open(output_file, "a+") as output:
			output.write(f'configs: max_new_tokens = {max_new_tokens} || temperature={temperature} || num_evidence = {num_evidence} || chunking = {chunking}')
	start = datetime.now()
	for index, question in questions.iterrows():
		number = str(question['No'])
		context = question['Context'].strip()
		actual_question = question['Question'].strip()

		llm_output, llm_input = get_answer(model, query, tokenizer, context, actual_question,
                                       max_new_tokens, temperature, num_evidence)
		with open(output_file, "a+") as output:
			output.write(f'\n\n\n Question {number} \n\n LLM Input:\n\n {llm_input} \n\n LLM Output:\n\n{llm_output}')

	end = datetime.now()
	total_time = str((end-start).total_seconds())
	with open(output_file, "a+") as output:
				output.write(f'\n\n\n Total Execution Time: {total_time} seconds')