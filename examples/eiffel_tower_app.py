"""
Interactive Eiffel Tower steering demo (Gradio).

Runs locally:  uv run examples/eiffel_tower_app.py
Requires:      uv sync --extra app   (installs gradio)

Shows baseline vs steered generation side by side for the same prompt and
seed, with a strength slider and generation-parameter controls.
"""

from loguru import logger

DATASET = "shawon/llama-3.1-8b-instruct_eiffel_tower"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


def respond(
    steering,
    prompt: str,
    strength_scale: float,
    temperature: float,
    repetition_penalty: float,
    seed: int,
):
    """Generate baseline and steered answers for the same prompt/seed."""
    common = dict(
        max_new_tokens=256,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        system_prompt=SYSTEM_PROMPT,
        seed=int(seed),
    )
    baseline = steering.generate(prompt, strength_scale=0.0, **common)
    steered = steering.generate(prompt, strength_scale=strength_scale, **common)
    return baseline, steered


def build_demo(steering):
    """Build the Gradio Blocks UI around a loaded SAESteering instance."""
    import gradio as gr

    fn = lambda prompt, scale, temp, rep, seed: respond(  # noqa: E731
        steering, prompt, scale, temp, rep, seed
    )

    with gr.Blocks(title="Eiffel Tower Llama (Drrik)") as demo:
        gr.Markdown(
            "# Eiffel Tower Llama\n"
            "SAE steering of Llama 3.1 8B Instruct with Anthropic's pre-trained "
            "features. Same prompt and seed on both sides — only the steering "
            "strength differs."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="Tell me about your favorite thing to do on a weekend.",
                lines=2,
            )
            with gr.Column():
                scale = gr.Slider(
                    0.0, 3.0, value=1.0, step=0.05, label="Strength scale"
                )
                temp = gr.Number(value=0.5, label="Temperature")
                rep = gr.Number(value=1.2, label="Repetition penalty")
                seed = gr.Number(value=16, label="Seed", precision=0)
                run = gr.Button("Generate", variant="primary")
        with gr.Row():
            baseline_out = gr.Textbox(label="Baseline (no steering)", lines=12)
            steered_out = gr.Textbox(label="Steered (Eiffel Tower)", lines=12)
        run.click(fn, [prompt, scale, temp, rep, seed], [baseline_out, steered_out])
    return demo


def main():
    from drrik import SAESteering
    from drrik.steering import SteeringVectors

    logger.info(f"Loading steering vectors from {DATASET}")
    vectors = SteeringVectors.from_hf_dataset(DATASET)
    logger.info(f"Loading {MODEL_NAME}")
    steering = SAESteering(source=vectors, model_name=MODEL_NAME)

    demo = build_demo(steering)
    demo.launch()


if __name__ == "__main__":
    main()
