import openai
import time
import asyncio
import os

#added this line for making sure we do not push the keys to github
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('/scratch/ltl2113/LLaVA-Med/.env')

# Use environment variables#so make sure you have a file .env in directory
openai.api_type = os.getenv("OPENAI_API_TYPE", "azure")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", 'https://example-endpoint.openai.azure.com/')
openai.api_version = os.getenv("OPENAI_API_VERSION", "2023-03-15-preview")
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "deployment-name")


async def dispatch_openai_requests(
  deployment_id,
  messages_list,
  temperature,
):
    async_responses = [
        openai.ChatCompletion.acreate(
            deployment_id=deployment_id,
            messages=x,
            temperature=temperature,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def call_async(samples, wrap_gen_message, print_result=False):
  message_list = []
  for sample in samples:
    input_msg = wrap_gen_message(sample)
    message_list.append(input_msg)
  
  try:
    predictions = asyncio.run(
      dispatch_openai_requests(
        deployment_id=DEPLOYMENT_ID,
        messages_list=message_list,
        temperature=0.0,
      )
    )
  except Exception as e:
    print(f"Error in call_async: {e}")
    time.sleep(6)
    return []

  results = []
  for sample, prediction in zip(samples, predictions):
    if prediction:
      if 'content' in prediction['choices'][0]['message']:
        sample['result'] = prediction['choices'][0]['message']['content']
        if print_result:
          print(sample['result'])
        results.append(sample)
  return results