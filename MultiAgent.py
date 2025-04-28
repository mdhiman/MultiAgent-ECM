import time
import os
import asyncio
from openai import AzureOpenAI
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

CHAT_COMPLETION_CLIENT = None
# Define the method to handle new change requests
def handle_new_change_request(new_entries):
    """
    Handle new entries from the monitored file.

    Args:
        new_entries (list): List of new lines added to the file.
    """
    for entry in new_entries:
        print(f"Handling new change request: {entry.strip()}")
        entry = entry.strip()  # Remove any trailing spaces or newlines
        if entry:
            items = entry.split(",")  # Split entry into items using comma as delimiter
            folder_name = items[0].strip()  # Use the first item as the folder name

            # Create a folder with the name of the first item
            try:
                os.makedirs(folder_name, exist_ok=True)
                print(f"Created folder: {folder_name}")
            except Exception as e:
                print(f"Error creating folder {folder_name}: {e}")
        # Process each new entry here
        try:
            proposal = asyncio.run(prepare_change_request_proposal(CHAT_COMPLETION_CLIENT,entry))
            proposal_file_path = os.path.join(folder_name, "draft_proposal.txt")

            # Save the proposal to a file
            with open(proposal_file_path, "w") as proposal_file:
                proposal_file.write(proposal)
            print(f"Draft proposal saved in: {proposal_file_path}")
        except Exception as e:
            print(f"Error generating draft proposal for {folder_name}: {e}")


# File monitoring function
def monitor_file(file_path):
    """
    Monitor a file for new entries and call the handler function when changes occur.

    Args:
        file_path (str): Path to the file to monitor.
    """
    print(f"Monitoring file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return

    with open(file_path, 'r') as file:
        # Move to the end of the file
        file.seek(0, os.SEEK_END)
        last_position = file.tell()

        while True:
            # Check for new data in the file
            current_position = file.tell()
            if current_position < last_position:
                # File was truncated; reset position
                print("File was truncated. Resetting position.")
                last_position = 0
                file.seek(0)

            line = file.readline()
            if line:
                # Process all new lines
                new_entries = [line]
                while True:
                    line = file.readline()
                    if not line:
                        break
                    new_entries.append(line)
                handle_new_change_request(new_entries)
                last_position = file.tell()

            # Sleep for a short period to avoid busy waiting
            time.sleep(1)


async def prepare_change_request_proposal(client, change_request):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Write proposal for the change and prepare a meanningful change request summary. Make it little short and highlight changes in bullet points.",
            },
            {
                "role": "user",
                "content": change_request,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model="gpt-4o"
    )

    return response.choices[0].message.content




async def main():
    # Initialize the kernel
    endpoint = "https://dhiman-test.openai.azure.com/"
    deployment = "gpt-4o"

    subscription_key = "your-api-key"
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    global CHAT_COMPLETION_CLIENT
    CHAT_COMPLETION_CLIENT=client


async def SemanticKernelChatCompletion():
    # Initialize the kernel
    kernel = Kernel()
    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="gpt-4o",
        api_key="your-api-key",
        endpoint="https://dhiman-test.openai.azure.com/"
    )
    kernel.add_service(chat_completion)

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel
        )

        # Print the results
        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)


# Entry point
if __name__ == "__main__":
    #asyncio.run(SemanticKernelChatCompletion())
    asyncio.run(main())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_to_monitor = os.path.join(script_dir,
                                   "change_requests.log")  # File relative to script location

    monitor_file(file_to_monitor)
