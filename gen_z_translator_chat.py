# Practice defining the function using a prompt string directly and adding a live chat bot
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Create a new kernel
kernel = Kernel()

# Set the service ID you'll use
service_id = "genz_service"

# Add Azure OpenAI 
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,
        env_file_path=".env",  
    )
)

# Get and customize the execution settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.9
req_settings.top_p = 0.9

# Define the Gen Z Translator semantic function
gen_z_translator = kernel.add_function(
    function_name="gen_z_translator",
    plugin_name="fun_translations",
    prompt="{{ $input }}\n\nNow translate that into Gen Z slang. Keep it fun, exaggerated, and up-to-date.",
    prompt_execution_settings=req_settings,
)

# Async main function
async def main():
    print("\n --- Gen Z Translator Chat ---")
    print("Type a sentence to translate to Gen Z slang.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("👵 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 🧃")
            break
        
        gen_zified = await kernel.invoke(gen_z_translator, input=user_input)
        print(f"🧃 Gen Z: {gen_zified}\n")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
