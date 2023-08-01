from utils import *
import time
import argparse
import sys
import json
from transformers import pipeline


def main():
    # load arguments from terminal
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    print(f"API_KEY: {API_KEY}")

    set_random_seed(args.random_seed)

    # load dataset
    dataloader = create_dataloader(args)

    #TODO1.2 新写了create_gpt_test_input_prompt函数，用于将选择出的qa对拼接成prompt
    if args.method == "few_shot":
        input_prompt = create_gpt_test_input_prompt(args)
    elif args.method == "few_shot_cot" or args.method == "auto_cot" or args.method == "active_cot":
        input_prompt = create_gpt_test_input_prompt(args)
    else:
        raise NotImplementedError
    
    #输出拼接好的prompt以供检查
    print('*****************************')
    print("prompt:"+"\n"+input_prompt)
    print('*****************************')

    start = time.time()
    print("Inference Start")
    if args.multipath != 1:
        print("Self-consistency Enabled, output each inference result is not available")

    # no limit on how many batches to inference, assume inference all batches
    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)
    
    correct, wrong_list, QA_record = inference_cot_gpt(args, dataloader, args.qes_limit, input_prompt)
    print(f"correct_num: {correct}")
    print(f"total: {args.qes_limit}")
    print(f"Accuracy: {correct / (args.qes_limit)}")
    end = time.time()
    print(f"Execution time: {end - start} seconds")

    print(f"wrong questions: {wrong_list}")

    #将同样的结果输出的summary这个文件中去
    with open("summary.txt", "w") as f:
        f.write(f"correct_num: {correct}\n")
        f.write(f"total: {args.qes_limit}\n")
        f.write(f"Accuracy: {correct / args.qes_limit}\n")
        end = time.time()
        f.write(f"Execution time: {end - start} seconds\n")
        f.write(f"wrong questions: {wrong_list}\n")
        
    # save the wrong predictions
    if args.output_dir is not None:
        path = f"{args.output_dir}/wrong_{args.dataset}.txt"
        orginal_stdout = sys.stdout
        with open(path, 'w') as f:
            sys.stdout = f
            for i in wrong_list:
                print(str(i))
        sys.stdout = orginal_stdout
        
        path = f"{args.output_dir}/QA_record_{args.dataset}.txt"
        with open(path, 'w') as f:
            f.write(json.dumps(QA_record, indent=4))

#实际上进行推理的函数，这里我没有调用源代码的gpt，而是选择了huggingface提供的pipeline，同时统计正确与错误的输出
def inference_cot(args, question_pool, qes_limit, given_prompt):
    correct = 0
    qes_count = 0
    wrong_list = []
    QA_record = []

    for qes_num, qes in enumerate(question_pool):
        if qes_limit is not None and qes_count == qes_limit:
            break
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []
        
        if args.dataset == "last_letters" and args.use_code_style_prompt is True:
            # code style prompt
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step in Python."
        else:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]

        # enable self-consistency if multipath > 1
        for path in range(0, args.multipath):
            model_name = "distilbert-base-cased-distilled-squad"
            model = pipeline("question-answering", model=model_name, tokenizer=model_name)
            prompt = "\n".join(prompt_list)
            responses = model(question=qes["question"],context=prompt)
            

            pred_ans = responses

            # create a dict to record each Q&A for later review purposes
            QA = {}
            QA['qes_idx'] = qes['question_idx']
            QA['Q'] = qes['question']
            QA['A'] = responses['answer']
            QA_record.append(QA)

            # output current inference result (only works when self-consistency is not enable)
            if args.multipath == 1:

                print('-' * 20)
                print(f"Question number: {qes_num}")
                print(f"Dataset index: {qes['question_idx']}")
                print(f"Q: " + qes['question'])
                if args.dataset == "last_letters" and args.use_code_style_prompt is True:
                    print(f"A: Let's think step by step in Python." + responses['answer'])
                else:
                    print(f"A: Let's think step by step." + responses['answer'])
                print(f"pred_ans: {pred_ans['answer']}")
                print(f"GT: {qes['answer']}")

            # record all answers into the self-consistency list to find the most frequent one
            all_self_consistency_ans.append(pred_ans)

        if qes["answer"] == pred_ans['answer']:
            correct += 1
        else:
            wrong_list.append({'idx':qes['question_idx'], 'pred_ans':pred_ans['answer'], 'GT':qes['answer']})

        qes_count += 1

    return correct, wrong_list, QA_record
def inference_cot_gpt(args, question_pool, qes_limit, given_prompt):
    correct = 0
    qes_count = 0
    wrong_list = []
    QA_record = []

    for qes_num, qes in enumerate(question_pool):
        if qes_limit is not None and qes_count == qes_limit:
            break
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []
        
        if args.dataset == "last_letters" and args.use_code_style_prompt is True:
            # code style prompt
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step in Python."
        else:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]

        # enable self-consistency if multipath > 1
        for path in range(0, args.multipath):
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, time_interval=args.api_time_interval,
                                      temperature=args.temperature, stop='\n')

            pred_ans = answer_extraction(args, responses)

            # create a dict to record each Q&A for later review purposes
            QA = {}
            QA['qes_idx'] = qes['question_idx']
            QA['Q'] = qes['question']
            QA['A'] = responses['choices'][0]['text']
            QA_record.append(QA)

            # output current inference result (only works when self-consistency is not enable)
            if args.multipath == 1:

                print('-' * 20)
                print(f"Question number: {qes_num}")
                print(f"Dataset index: {qes['question_idx']}")
                print(f"Q: " + qes['question'])
                if args.dataset == "last_letters" and args.use_code_style_prompt is True:
                    print(f"A: Let's think step by step in Python." + responses['choices'][0]['text'])
                else:
                    print(f"A: Let's think step by step." + responses['choices'][0]['text'])
                print(f"pred_ans: {pred_ans}")
                print(f"GT: {qes['answer']}")

            # record all answers into the self-consistency list to find the most frequent one
            all_self_consistency_ans.append(pred_ans)

        if pred_ans== qes['answer']:
            correct += 1
        else:
            wrong_list.append({'idx':qes['question_idx'], 'pred_ans':pred_ans, 'GT':qes['answer']})

        qes_count += 1

    return correct, wrong_list, QA_record

#此处配置了各项信息，类似于写了一个shell命令
def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    #我的做法是在得到的result中提取序号，再回到数据集中去提取问题拼接为prompt，所以这里需要两个路径
    parser.add_argument(
        "--prompt_source_path", type=str, default="./dataset/GSM8K/train.jsonl", help="prompts to use"
    )
    parser.add_argument(
        "--prompt_num_path", type=str, default="./logdifference_results/logdifference_result_gsm8k_from_70_questions.txt", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="text-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="active_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=50, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    
    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")
    
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.dataset_path = "./dataset/ASDiv/ASDiv.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "The answer is"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "So the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/MAWPS/SingleEq.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MAWPS/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


if __name__ == "__main__":
    main()