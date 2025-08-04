from .utils import args_exp_parser, eval_config_printer
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Eval():
    def __init__(self, args):

        # args for inference
        self.model = args_exp_parser(args, "model")
        self.is_reasoning = args_exp_parser(args, "is_reasoning")
        self.tasks = args_exp_parser(args, "tasks")
        self.device = args_exp_parser(args, "device")
        self.batch_size = args_exp_parser(args, "batch_size")
        self.tensor_parallel = args_exp_parser(args, "tensor_parallel")
        self.temperature = args_exp_parser(args, "temperature")
        self.top_k = args_exp_parser(args, "top_k")
        self.top_p = args_exp_parser(args, "top_p")
        self.max_tokens = args_exp_parser(args, "max_tokens")
        self.seed = args_exp_parser(args, "seed")
        
        # args for etc
        self.setproctitle = args_exp_parser(args, "setproctitle")
        self.debug = args_exp_parser(args, "debug")

        # args for output
        self.eval_type = args_exp_parser(args, "eval_type")
        self.output_dir = args_exp_parser(args, "output_dir")
        self.save_logs = args_exp_parser(args, "save_logs")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def __call__(self):
        eval_config_printer(self)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        template = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}]
        tokenized_template = self.tokenizer.apply_chat_template(template, tokenize=False)
        sampling_params = SamplingParams(
                                        temperature=self.temperature,
                                        top_k=self.top_k,
                                        top_p=self.top_p,
                                        max_tokens=self.max_tokens,
                                        seed=self.seed,
                                        )
        if not self.tensor_parallel:
            llm = LLM(model=self.model)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.device)
            llm = LLM(model=self.model, tensor_parallel_size=len(self.device))
        outputs = llm.generate(tokenized_template, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

class DataLoader():
    def __init__(self, tasks, batch_size):
        self.tasks = tasks
        self.batch_size = batch_size

    def load_data(self):
        # Placeholder for data loading logic
        print(f"Loading data for tasks: {self.tasks} with batch size: {self.batch_size}")
        return ["Sample data"] * len(self.tasks)  # Dummy data