from typing import List, Union

# function which parses command line arguments for evaluation
def args_exp_parser(args, arg_name) -> Union[str, List[str], int]:
    if arg_name == "model":
        result: str = args.model
        return result

    elif arg_name == "tasks":
        if "," in args.tasks:
            parsed_tasks: List[str] = []
            tasks = args.tasks.split(",")
            for task in tasks:
                if task != " ":
                    parsed_tasks.append(task.strip())
            return parsed_tasks
        else:
            result: List[str] = [args.tasks.strip()]
            return result

    elif arg_name == "device":
        if "," in args.device:
            parsed_devices: List[str] = []
            devices = args.device.split(",")
            for device in devices:
                if device != " ":
                    if device.isdecimal():
                        parsed_devices.append(device.strip())
                    else:
                        raise ValueError(f"Invalid device argument: {device}")
            return parsed_devices
        else:
            result: List[str] = [args.device.strip()]
            return result

    elif arg_name == "batch_size":
        if str(args.batch_size).isdecimal():
            result: int = int(args.batch_size)
            return result
        else:
            raise ValueError(f"Invalid batch size argument: {args.batch_size}")

    elif arg_name == "eval_type":
        # Should implement the case of generation with multi answers 
        if args.eval_type in ["generation", "logit"]:
            result: str = args.eval_type
            return result
        else:
            raise ValueError(f"Invalid eval type argument: {args.eval_type}")
        
    elif arg_name == "is_reasoning":
        if isinstance(args.is_reasoning, bool):
            return args.is_reasoning
        else:
            raise ValueError(f"Invalid is_reasoning argument: {args.is_reasoning}")
    
    elif arg_name == "setproctitle":
        if isinstance(args.setproctitle, str):
            return args.setproctitle
        else:
            raise ValueError(f"Invalid setproctitle argument: {args.setproctitle}")
    
    elif arg_name == "output_dir":
        if isinstance(args.output_dir, str):
            return args.output_dir
        else:
            raise ValueError(f"Invalid output_dir argument: {args.output_dir}")
    
    elif arg_name == "debug":
        if isinstance(args.debug, bool):
            return args.debug
        else:
            raise ValueError(f"Invalid debug argument: {args.debug}")
    
    elif arg_name == "save_logs":
        if isinstance(args.save_logs, bool):
            return args.save_logs
        else:
            raise ValueError(f"Invalid save_logs argument: {args.save_logs}")
    
    elif arg_name == "tensor_parallel":
        if isinstance(args.tensor_parallel, bool):
            if args.tensor_parallel and "," not in args.device:
                raise ValueError("Tensor parallelism requires multiple devices.")
            return args.tensor_parallel
        else:
            raise ValueError(f"Invalid tensor_parallel argument: {args.tensor_parallel}")
        
    elif arg_name == "temperature":
        if isinstance(args.temperature, float):
            return args.temperature
        else:
            raise ValueError(f"Invalid temperature argument: {args.temperature}")
        
    elif arg_name == "top_p":
        if isinstance(args.top_p, float):
            return args.top_p
        else:
            raise ValueError(f"Invalid top_p argument: {args.top_p}")
        
    elif arg_name == "top_k":
        if isinstance(args.top_k, int):
            return args.top_k
        else:
            raise ValueError(f"Invalid top_k argument: {args.top_k}")
        
    elif arg_name == "max_tokens":
        if isinstance(args.max_tokens, int):
            return args.max_tokens
        else:
            raise ValueError(f"Invalid max_tokens argument: {args.max_tokens}")
        
    elif arg_name == "seed":
        if isinstance(args.seed, int):
            return args.seed
        else:
            raise ValueError(f"Invalid seed argument: {args.seed}")
    
    else:
        raise ValueError(f"Not Implemented: {arg_name}")

# function which convert string to boolean, especially for argparse library paradox
def str2bool(v):
    if isinstance(v, bool):
        return v
    
    if v.lower() in ('true', 'True'):
        return True
    
    elif v.lower() in ('false', 'False'):
        return False

# function which prints the evaluation configuration while running the evaluation
def eval_config_printer(args):
    print("=============================================")
    print("             Eval Configuration")
    print("---------------------------------------------")
    if args.is_reasoning:
        print(f"       Reasoning Mode for hybrid model")
        print("---------------------------------------------")
    if args.debug:
        print("                 Debug Mode")
    if args.debug:
        print("---------------------------------------------")
    print(f"Process Title: {args.setproctitle}")
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Device: {', '.join(args.device)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Eval Type: {args.eval_type}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Save Logs: {args.save_logs}")
    print("=============================================")
