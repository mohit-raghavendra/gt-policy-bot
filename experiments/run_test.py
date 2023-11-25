import pandas as pd
from datetime import datetime
import numpy as np
import torch


def get_answer_with_evidence(
    model,
    query,
    tokenizer,
    context,
    question,
    max_new_tokens,
    temperature,
    num_evidence,
    evidence=None,
):
    full_question = context + " " + question
    baseline_context = "You are student at Georgia Institue of Technology and you have a question about its legal policies."
    if context == baseline_context:
        evidences = query.run_query(question, num_evidence)
    else:
        evidences = query.run_query(full_question, num_evidence)
    all_evidence = ""
    i = 0
    for evidence in evidences:
        all_evidence += str("Evidence " + str(i) + ": " + evidence + " ")
        i += 1
    full_context = f"Give a precise answer to the question based on only the context and evidence and do not be verbose. ### Context:{full_question}\n ### Evidence: {all_evidence}\n ### Answer:"
    inputs = tokenizer(full_context, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=3,
        return_dict_in_generate=True,
        output_scores=True,
    )
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:].cpu()
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    transition_scores = transition_scores.cpu()
    max_len = max([len(generated_tokens[i]) for i in range(generated_tokens.shape[0])])
    hal_dist = 0
    h_t = 0
    print(generated_tokens.shape[0])
    for i in range(max_len):
        probs = []
        for j in range(generated_tokens.shape[0]):
            if i >= len(generated_tokens[j]):
                continue
            tok = generated_tokens[j][i]
            score = transition_scores[j][i]
            # | token | token string | logits | probability
            print(
                f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}"
            )
            # output += tokenizer.decode(tok)
            prob = np.exp(score.numpy())
            probs.append(prob)

        flag = True
        for prob in probs:
            if not (prob <= 0.51 and prob > 0.0):
                flag = False
                break
        if flag:
            h_t += 1
            print((max(probs) - min(probs)))
            hal_dist += 1 - abs(max(probs) - min(probs)) - 0.00000000000000001 + h_t
        print(hal_dist)
    del transition_scores
    del outputs
    torch.cuda.empty_cache()
    full_context += "\nHallucination Distance: " + str(hal_dist)
    return "", full_context


def get_answer_without_evidence(model, tokenizer, context, question):
    full_question = context + " " + question
    full_context = f"Give a precise answer to the question and do not be verbose. Context: {full_question}\n Answer:"
    inputs = tokenizer(full_context, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        return_dict_in_generate=True,
        num_return_sequences=3,
        output_scores=True,
    )
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:].cpu()
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    transition_scores = transition_scores.cpu()
    max_len = max([len(generated_tokens[i]) for i in range(generated_tokens.shape[0])])
    hal_dist = 0
    h_t = 0
    for i in range(max_len):
        probs = []
        for j in range(generated_tokens.shape[0]):
            if i >= len(generated_tokens[j]):
                continue
            tok = generated_tokens[j][i]
            score = transition_scores[j][i]
            # | token | token string | logits | probability
            print(
                f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}"
            )
            # output += tokenizer.decode(tok)
            prob = np.exp(score.numpy())
            probs.append(prob)

        flag = True
        for prob in probs:
            if not (prob <= 0.51 and prob > 0.0):
                flag = False
                break
        if flag:
            h_t += 1
            print((max(probs) - min(probs)))
            hal_dist += 1 - abs(max(probs) - min(probs)) - 0.00000000000000001 + h_t
        print(hal_dist)
    del transition_scores
    del outputs
    torch.cuda.empty_cache()
    full_context += "\nHallucination Distance: " + str(hal_dist)
    return "", full_context


def run_test(test, model, query, tokenizer, with_evidence=True):
    questions = pd.read_csv("data/code_of_conduct/questions.csv")
    # make sure to change output file name each time and download locally
    output_file = test[0]
    # output configurations to top of file
    max_new_tokens = test[1]
    temperature = test[2]
    num_evidence = test[3]
    chunking = test[4]

    with open(output_file, "a") as output:
        output.write(
            f"configs: max_new_tokens = {max_new_tokens} || temperature={temperature} || num_evidence = {num_evidence} || chunking = {chunking}"
        )
    start = datetime.now()
    for index, question in questions.iterrows():
        number = str(question["No"])
        context = question["Context"].strip()
        actual_question = question["Question"].strip()

        if with_evidence:
            llm_output, llm_input = get_answer_with_evidence(
                model,
                query,
                tokenizer,
                context,
                actual_question,
                max_new_tokens,
                temperature,
                num_evidence,
            )
        else:
            llm_output, llm_input = get_answer_without_evidence(
                model, tokenizer, context, actual_question
            )
        with open(output_file, "a") as output:
            output.write(
                f"\n\n\n Question {number} \n\n LLM Input:\n\n {llm_input} \n\n LLM Output:\n\n{llm_output}"
            )

    end = datetime.now()
    total_time = str((end - start).total_seconds())
    with open(output_file, "a") as output:
        output.write(f"\n\n\n Total Execution Time: {total_time} seconds")
