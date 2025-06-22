import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Initialize kernel and service
kernel = Kernel()
service_id = "lie_service"

# Register Azure OpenAI service from .env
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,
        env_file_path=".env",
    )
)

# Define request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 200
req_settings.temperature = 0.2
req_settings.top_p = 0.9

# Define the prompt template for the lie detector
lie_detector_prompt = """
Statement: {{ $input }}

Is the above statement likely to be true or false? 
Respond with "True" or "False" and a short explanation.
"""

# Wrap it in PromptTemplateConfig
lie_template_config = PromptTemplateConfig(
    template=lie_detector_prompt,
    name="lie_detector",
    template_format="semantic-kernel",
    execution_settings=req_settings,
)

# Add the semantic function to the kernel
lie_detector = kernel.add_function(
    function_name="lie_detector",
    plugin_name="truth_teller",
    prompt_template_config=lie_template_config,
)

# Main async function
async def main():
    print("\n--- Lie Detector Interactive ---")
    print("Type a statement to check if it’s likely true or false.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("❓ Enter your statement: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        verdict = await kernel.invoke(lie_detector, input=user_input)
        print(f"🔍 Verdict: {verdict}\n")


# Run it
if __name__ == "__main__":
    asyncio.run(main())
