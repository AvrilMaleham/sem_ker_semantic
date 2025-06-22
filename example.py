import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

kernel = Kernel()
service_id="test"

# Prepare AzureOpenAI service using credentials stored in the `.env` file
kernel.add_service(
    AzureChatCompletion(
        service_id=service_id,  
        env_file_path=".env",
    )
)

# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8

prompt = """
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words."""

# Create a prompt template configuration

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    execution_settings=req_settings,
)

# Add the function to the kernel using the prompt template configuration
# In this case, the function will use the prompt template to generate a response (all in one as a simple demo).

function = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt_template_config=prompt_template_config,
)


# -------------------------------------------------------------------------------------------------------------- #
# Below is an alternative way to define the function using a prompt string directly
# This is useful for quick tests or simple functions
# Note: This will not use the prompt template config, but the prompt string directly.

summarize = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt="{{$input}}\n\nOne line TLDR with the fewest words.",
    prompt_execution_settings=req_settings,
)

# Run your prompt
# Note: functions are run asynchronously
async def main():
    print("\n --- Running TLDR function --- ")
    result = await kernel.invoke(function)
    print(result) # => Robots must not harm humans.

    print("\n --- Summarizing Laws of Thermodynamics and Motion (New 'function' with an 'input') ---")

    print(await kernel.invoke(summarize, input="""
    1st Law of Thermodynamics - Energy cannot be created or destroyed.
    2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
    3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""))

    # Summarize the laws of motion
    print(await kernel.invoke(summarize, input="""
    1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
    2. The acceleration of an object depends on the mass of the object and the amount of force applied.
    3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first."""))

    # Summarize the law of universal gravitation
    print(await kernel.invoke(summarize, input="""
    Every point mass attracts every single other point mass by a force acting along the line intersecting both points.
    The force is proportional to the product of the two masses and inversely proportional to the square of the distance between them."""))


if __name__ == "__main__":
    asyncio.run(main())