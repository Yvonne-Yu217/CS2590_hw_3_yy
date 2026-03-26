import argparse
import gc
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExperimentSummary:
    name: str
    model_id: str
    prompt_prefix: str
    prompt_suffix: str
    config: dict
    n_sample: int
    lower_bound: int
    upper_bound: int
    acc: float
    mae: float
    res: float
    prompt_length: int


def dprint(msg: str, debug: bool):
    if debug:
        print(msg)


def load_model(model_id: str, device: str):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    # Prefer reduced precision on GPU to avoid OOM on 7B/8B models.
    model_kwargs = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_addition_pairs(lower_bound: int, upper_bound: int, rng: np.random.Generator):
    int_a = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    int_b = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    return int_a, int_b


def postproc_digits_anywhere(output_string: str) -> int:
    only_digits = re.sub(r"\D", "", output_string)
    try:
        return int(only_digits)
    except Exception:
        return 0


def postproc_answer_first_line(output_string: str) -> int:
    first_line = output_string.strip().splitlines()[0] if output_string.strip() else ""

    # Prefer Answer-labeled number first.
    m_ans = re.search(r"answer\s*[:=]\s*([-+]?\d+)", first_line, flags=re.IGNORECASE)
    if m_ans:
        try:
            return int(m_ans.group(1))
        except Exception:
            pass

    m_any = re.search(r"[-+]?\d+", first_line)
    if m_any:
        try:
            return int(m_any.group(0))
        except Exception:
            pass

    return postproc_digits_anywhere(output_string)


def call_model(prompt: str, student_configs: dict, post_processing_fn, model, tokenizer, device: str, debug: bool = False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hf_configs = student_configs.copy()
    if "max_tokens" in hf_configs:
        hf_configs["max_new_tokens"] = hf_configs.pop("max_tokens")
    hf_configs.pop("stop", None)
    hf_configs.setdefault("pad_token_id", tokenizer.eos_token_id)

    outputs = model.generate(**inputs, **hf_configs)
    result_new = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    dprint("************ Prompt ************", debug)
    dprint(prompt, debug)
    dprint("\n************ Raw Response ************", debug)
    dprint(result_new, debug)

    final_output = post_processing_fn(result_new)

    dprint("\n************ Final Output ************", debug)
    dprint(str(final_output), debug)

    return final_output


def test_range(
    name: str,
    added_prompt,
    prompt_configs,
    rng,
    model,
    tokenizer,
    device,
    n_sample=10,
    lower_bound=1,
    upper_bound=10,
    fixed_pairs=None,
    pre_processing=lambda x: x,
    post_processing=postproc_digits_anywhere,
    debug=False,
    sleep_sec=0.0,
):
    int_as = []
    int_bs = []
    answers = []
    model_responses = []
    correct = []
    prompts = []

    iterations = list(fixed_pairs) if fixed_pairs is not None else list(range(n_sample))

    for v in iterations:
        if fixed_pairs is None:
            int_a, int_b = get_addition_pairs(lower_bound, upper_bound, rng)
        else:
            int_a, int_b = v

        fixed_prompt = pre_processing(f"{int_a}+{int_b}")
        prefix, suffix = added_prompt
        prompt = prefix + fixed_prompt + suffix

        model_response = call_model(prompt, prompt_configs, post_processing, model, tokenizer, device, debug=debug)
        answer = int_a + int_b

        int_as.append(int_a)
        int_bs.append(int_b)
        prompts.append(prompt)
        answers.append(answer)
        model_responses.append(model_response)
        correct.append(answer == model_response)

        if sleep_sec > 0:
            sleep(sleep_sec)

    df = pd.DataFrame(
        {
            "int_a": int_as,
            "int_b": int_bs,
            "prompt": prompts,
            "answer": answers,
            "response": model_responses,
            "correct": correct,
        }
    )

    mae = float(mean_absolute_error(df["answer"], df["response"]))
    acc = float(df.correct.sum() / len(df))
    prompt_length = len(prefix) + len(suffix)
    res = float(acc * (1 / prompt_length) * (1 - mae / (1 * 10**4)))

    return df, {"res": res, "acc": acc, "mae": mae, "prompt_length": prompt_length}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_report_tex(results: dict, out_tex: Path):
    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\title{Prompt Engineering for Addition: Experiment Report}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\section*{Run Metadata}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item Timestamp: {results['timestamp']}")
    lines.append(rf"\item Device: {results['device']}")
    lines.append(r"\end{itemize}")
    lines.append(r"\section*{Experiment Metrics}")
    lines.append(r"\begin{longtable}{lllllll}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & Model & Acc & MAE & Res & PromptLen & Samples \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for item in results["summaries"]:
        lines.append(
            f"{item['name']} & {item['model_id']} & {item['acc']:.4f} & {item['mae']:.4f} & {item['res']:.6f} & {item['prompt_length']} & {item['n_sample']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    lines.append(r"\section*{Notes}")
    lines.append(r"This file is auto-generated from run\_prompting\_experiments.py.")
    lines.append(r"Use the JSON/CSV outputs for detailed row-level inspection.")
    lines.append(r"\end{document}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def run_experiment_and_record(
    results_store: dict,
    out_dir: Path,
    name: str,
    model,
    tokenizer,
    model_id: str,
    added_prompt,
    prompt_config,
    rng,
    device,
    n_sample,
    lower_bound,
    upper_bound,
    post_processing,
    fixed_pairs=None,
    debug=False,
):
    df, metrics = test_range(
        name=name,
        added_prompt=added_prompt,
        prompt_configs=prompt_config,
        rng=rng,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_sample=n_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_pairs=fixed_pairs,
        pre_processing=lambda x: x,
        post_processing=post_processing,
        debug=debug,
        sleep_sec=0.0,
    )

    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    summary = ExperimentSummary(
        name=name,
        model_id=model_id,
        prompt_prefix=added_prompt[0],
        prompt_suffix=added_prompt[1],
        config=prompt_config,
        n_sample=len(df),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        acc=metrics["acc"],
        mae=metrics["mae"],
        res=metrics["res"],
        prompt_length=metrics["prompt_length"],
    )
    results_store["summaries"].append(asdict(summary))
    results_store["files"].append(str(csv_path))

    print(f"[{name}] {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Run prompting_exercises.ipynb experiments in script form.")
    parser.add_argument("--llama-model", default=os.getenv("LLAMA_LOCAL_MODEL_PATH", "meta-llama/Llama-2-7b-chat-hf"))
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen experiments if resources are limited.")
    parser.add_argument("--skip-part2", action="store_true", help="Skip Part 2 (Q4*) experiments.")
    parser.add_argument("--n-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/prompting_experiments")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rng = np.random.default_rng(args.seed)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "seed": args.seed,
        "summaries": [],
        "files": [],
    }

    # Load Llama (or local equivalent)
    model, tokenizer = load_model(args.llama_model, device)

    # Q3 zero-shot setup
    added_prompt = ("Question: What is ", "?\\nAnswer: ")
    prompt_config = {
        "max_tokens": 2,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.6,
        "repetition_penalty": 1,
        "stop": [],
    }

    run_experiment_and_record(
        results,
        out_dir,
        "q3_zero_shot_1digit",
        model,
        tokenizer,
        args.llama_model,
        added_prompt,
        prompt_config.copy(),
        rng,
        device,
        args.n_sample,
        1,
        10,
        postproc_digits_anywhere,
        debug=args.debug,
    )

    prompt_config_7 = prompt_config.copy()
    prompt_config_7["max_tokens"] = 8
    run_experiment_and_record(
        results,
        out_dir,
        "q3_zero_shot_7digit",
        model,
        tokenizer,
        args.llama_model,
        added_prompt,
        prompt_config_7.copy(),
        rng,
        device,
        args.n_sample,
        1000000,
        9999999,
        postproc_digits_anywhere,
        debug=args.debug,
    )

    # Q4 Part 2 experiments on Llama
    if not args.skip_part2:
        added_prompt_q4 = ("Question: What is 3+7?\\nAnswer: 10\\n Question: What is ", "?\\nAnswer: ")
        prompt_config_q4 = {
            "max_tokens": 8,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.6,
            "repetition_penalty": 1,
            "stop": [],
        }

        run_experiment_and_record(
            results,
            out_dir,
            "q4a_one_shot_1digit_example_max8",
            model,
            tokenizer,
            args.llama_model,
            added_prompt_q4,
            prompt_config_q4.copy(),
            rng,
            device,
            args.n_sample,
            1000000,
            9999999,
            postproc_digits_anywhere,
            debug=args.debug,
        )

        prompt_config_q4b = prompt_config_q4.copy()
        prompt_config_q4b["max_tokens"] = 50
        run_experiment_and_record(
            results,
            out_dir,
            "q4b_one_shot_1digit_example_max50",
            model,
            tokenizer,
            args.llama_model,
            added_prompt_q4,
            prompt_config_q4b.copy(),
            rng,
            device,
            args.n_sample,
            1000000,
            9999999,
            postproc_answer_first_line,
            debug=args.debug,
        )

        added_prompt_q4c = (
            "Question: What is 1234567+1234567?\\nAnswer: 2469134\\nQuestion: What is ",
            "?\\nAnswer: ",
        )

        prompt_config_q4c8 = prompt_config_q4.copy()
        prompt_config_q4c8["max_tokens"] = 8
        run_experiment_and_record(
            results,
            out_dir,
            "q4c_in_distribution_max8",
            model,
            tokenizer,
            args.llama_model,
            added_prompt_q4c,
            prompt_config_q4c8.copy(),
            rng,
            device,
            args.n_sample,
            1000000,
            9999999,
            postproc_answer_first_line,
            debug=args.debug,
        )

        prompt_config_q4c50 = prompt_config_q4.copy()
        prompt_config_q4c50["max_tokens"] = 50
        run_experiment_and_record(
            results,
            out_dir,
            "q4c_in_distribution_max50",
            model,
            tokenizer,
            args.llama_model,
            added_prompt_q4c,
            prompt_config_q4c50.copy(),
            rng,
            device,
            args.n_sample,
            1000000,
            9999999,
            postproc_answer_first_line,
            debug=args.debug,
        )

        # Q4d fixed pair repeated 5 times to observe variance
        for i in range(5):
            run_experiment_and_record(
                results,
                out_dir,
                f"q4d_fixed_pair_run_{i + 1}",
                model,
                tokenizer,
                args.llama_model,
                added_prompt_q4c,
                prompt_config_q4c50.copy(),
                rng,
                device,
                args.n_sample,
                1000000,
                9999999,
                postproc_answer_first_line,
                fixed_pairs=[(9090909, 1010101)],
                debug=args.debug,
            )

    unload_model(model, tokenizer)

    # Q3c and notebook Q4d(max_tokens=20) style experiment on Qwen
    if not args.skip_qwen:
        try:
            model_2, tokenizer_2 = load_model(args.qwen_model, device)

            prompt_config_qwen8 = {
                "max_tokens": 8,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.6,
                "repetition_penalty": 1,
                "stop": [],
            }
            run_experiment_and_record(
                results,
                out_dir,
                "q3c_qwen_7digit_max8",
                model_2,
                tokenizer_2,
                args.qwen_model,
                added_prompt,
                prompt_config_qwen8.copy(),
                rng,
                device,
                args.n_sample,
                1000000,
                9999999,
                postproc_digits_anywhere,
                debug=args.debug,
            )

            prompt_config_qwen20 = prompt_config_qwen8.copy()
            prompt_config_qwen20["max_tokens"] = 20
            run_experiment_and_record(
                results,
                out_dir,
                "q3d_qwen_7digit_max20",
                model_2,
                tokenizer_2,
                args.qwen_model,
                added_prompt,
                prompt_config_qwen20.copy(),
                rng,
                device,
                args.n_sample,
                1000000,
                9999999,
                postproc_digits_anywhere,
                debug=args.debug,
            )

            unload_model(model_2, tokenizer_2)
        except torch.OutOfMemoryError:
            print("Warning: OOM while running Qwen experiments. Skipping Qwen section.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    tex_path = out_dir / "report.tex"
    write_report_tex(results, tex_path)

    print(f"Wrote metrics JSON: {json_path}")
    print(f"Wrote LaTeX report: {tex_path}")
    print("To compile PDF on HPC: pdflatex -output-directory {} {}".format(out_dir, tex_path))


if __name__ == "__main__":
    main()
