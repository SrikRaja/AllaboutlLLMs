# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:29:35 2024

@author: Administrator
"""

import os

HF_API_KEY = os.environ.get("HF_API_KEY")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
# from langchain_openai import AzureChatOpenAI <- if you are running on Azure you might need this
# refer to this for Azure Chat Open AI: https://python.langchain.com/docs/integrations/chat/azure_chat_openai/
chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

prompt = """Explain what is Generative AI?"""
print(prompt)

response = chatgpt.invoke(prompt)
response

response.content

from langchain_community.llms import HuggingFaceEndpoint

MISTRAL7B_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
mistral_params = {
                  "wait_for_model": True,
                  "do_sample": False,
                  "return_full_text": False,
                  "max_new_tokens": 1000,
                }
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token = HF_API_KEY,
    endpoint_url=MISTRAL7B_API_URL,
    task="text-generation",
    **mistral_params
)


llm.invoke(prompt)

from langchain_community.chat_models import ChatHuggingFace

hf_mistral = ChatHuggingFace(llm=llm, model_id='mistralai/Mistral-7B-Instruct-v0.2', token = HF_API_KEY)

response = hf_mistral.invoke(prompt)
response

print(response.content)

chatgpt



from langchain_core.messages import HumanMessage, SystemMessage

prompt = """Can you explain what is Generative AI in 3 bullet points?"""

messages = [
    SystemMessage(content="Act as a helpful assistant."),
    HumanMessage(content=prompt),
]

messages

response = chatgpt.invoke(messages)
response

print(response.content)

#MEMORIZE THE CONVERSATION 
messages.append(response)

prompt = """What did we discuss so far?"""
messages.append(HumanMessage(content=prompt))
messages

response = chatgpt.invoke(messages)
response.content



# not needed if you are only running chatgpt
# this runs prompts using the open source LLM - however mistral doesnt support a system prompt
prompt = """Can you explain what is Generative AI in 3 bullet points?"""
messages = [
    HumanMessage(content=prompt),
]

response = hf_mistral.invoke(messages) # MISTRAL doesn't support system prompts
print(response.content)
messages.append(response)


prompt = """What did we discuss so far?"""
messages.append(HumanMessage(content=prompt))


response = hf_mistral.invoke(messages) # MISTRAL doesn't support system prompts
print(response.content)


messages.append(response)
prompt = """Summarize what has been discussed so far in one line"""
messages.append(HumanMessage(content=prompt))
response = hf_mistral.invoke(messages) # MISTRAL doesn't support system prompts
print(response.content)
messages.append(response)

messages.append(HumanMessage(content=prompt))



####################################
# more complex prompt with placeholders
from langchain.prompts import PromptTemplate

# Simple prompt

prompt = """Explain to me what is Generative AI in 3 bullet points?"""
prompt_template = PromptTemplate.from_template(prompt)
prompt_template
prompt = """Explain to me briefly about {topic} in {language}."""

prompt_template = PromptTemplate.from_template(prompt)
prompt_template

inputs = [("Artificial Intelligence", "english"),
          ("Artificial Intelligence", "telugu")]

prompts = [prompt_template.format(topic=topic, language=language) for topic, language in inputs]
prompts

responses = chatgpt.map().invoke(prompts)

for response in responses:
  print(response.content)
  print('-----')
  
  
from langchain_core.prompts import ChatPromptTemplate

# more complex prompt with placeholders
prompt = """Explain to me briefly about {topic}."""

chat_template = ChatPromptTemplate.from_template(prompt)
chat_template

topics = ['Generative AI', 'Machine Learning', 'Deep Learning']
prompts = [chat_template.format(topic=topic) for topic in topics]
prompts

responses = chatgpt.map().invoke(prompts)
for response in responses:
  print(response.content)
  print('-----')
  
  
# dont use if you are only using chatgpt
responses = hf_mistral.map().invoke(prompts)
for response in responses:
  print(response.content)
  print('-----')


messages = [
        ("system", "Act as an expert in real estate and provide brief answers"),
        ("human", "what is your name?"),
        ("ai", "my name is AIBot"),
        ("human", "{user_prompt}"),
]
chat_template = ChatPromptTemplate.from_messages(messages)
chat_template


text_prompts = ["what is your name?",
                "explain commercial real estate to me"]
chat_prompts = [chat_template.format(user_prompt=prompt) for prompt in text_prompts]
chat_prompts

print(chat_prompts[1])

responses = chatgpt.map().invoke(chat_prompts)
for response in responses:
  print(response.content)
  print('-----')
  
  
messages = [
        ("system", "Act as an expert in real estate and provide very detailed answers with examples"),
        ("user", "what is your name?"),
        ("ai", "my name is Junk John"),
        ("user", "{user_prompt}"),
]
chat_template = ChatPromptTemplate.from_messages(messages)
text_prompts = ["what is your name?"]
chat_prompts = [chat_template.format(user_prompt=prompt) for prompt in text_prompts]
chat_prompts

responses = chatgpt.map().invoke(chat_prompts)
for response in responses:
  print(response.content)
  print('-----')
  
  
  
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# Define your desired data structure - like a python data class.
class ITSupportResponse(BaseModel):
    orig_msg: str = Field(description="The original customer IT support query message")
    orig_lang: str = Field(description="Detected language of the customer message e.g. Spanish")
    category: str = Field(description="1-2 word describing the category of the problem")
    trans_msg: str = Field(description="Translated customer IT support query message in English")
    response: str = Field(description="Response to the customer in their original language - orig_lang")
    trans_response: str = Field(description="Response to the customer in English")


parser = JsonOutputParser(pydantic_object=ITSupportResponse)
parser

print(parser.get_format_instructions())


# And a query intented to prompt a language model to populate the data structure.
prompt_txt = """
             Act as an Information Technology (IT) customer support agent. For the IT support message mentioned below
             in triple backticks use the following format when generating the output response

             Output format instructions:
             {format_instructions}

             Customer IT support message:
             ```{it_support_msg}```
             """


# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=ITSupportResponse)

prompt = PromptTemplate.from_template(template=prompt_txt)
prompt


# create a simple LLM Chain - more on this later - uses LCEL - langchain expression language
llm_chain = (prompt
              |
            chatgpt
              |
            parser)
llm_chain


it_support_queue = [
    "Não consigo sincronizar meus contatos com o telefone. Sempre recebo uma mensagem de falha.",
    "Ho problemi a stampare i documenti da remoto. Il lavoro non viene inviato alla stampante di rete.",
    "プリンターのトナーを交換しましたが、印刷品質が低下しています。サポートが必要です。",
    "Я не могу войти в систему учета времени, появляется сообщение об ошибке. Мне нужна помощь.",
    "Internet bağlantım çok yavaş ve bazen tamamen kesiliyor. Yardım eder misiniz?",
    "Не могу установить обновление безопасности. Появляется код ошибки. Помогите, пожалуйста."
]

formatted_msgs = [{"it_support_msg": msg, "format_instructions": parser.get_format_instructions()}
                    for msg in it_support_queue]
formatted_msgs[0]



responses = llm_chain.map().invoke(formatted_msgs)


responses[0], type(responses[0])


import pandas as pd

df = pd.DataFrame(responses)
df



##IMPLEMENTING CACHING 
# integrations with other caching tools:
# https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.cache
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache


from datetime import datetime
start_time = datetime.now()


set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer

from langchain_core.prompts import ChatPromptTemplate

prompt = """Explain to me what is writ petition"""

chat_template = ChatPromptTemplate.from_template(prompt)

chatgpt.invoke(chat_template.format())


# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

  
  


start_time = datetime.now()


# The second time it is, so it goes faster
chatgpt.invoke(chat_template.format())

# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="langchain.db"))


start_time = datetime.now()

# The first time, it is not yet in cache, so it should take longer
prompt = """Explain to me what are Maximum Entropy methods"""

chat_template = ChatPromptTemplate.from_template(prompt)

chatgpt.invoke(chat_template.format())

# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))



start_time = datetime.now()



chat_template = ChatPromptTemplate.from_template(prompt)

chatgpt.invoke(chat_template.format())

# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


#streaming in LLMs - Runnable interface enables streaming 
prompt = """Explain to me what are fundamental rights and duties and differences as per Indian constitution"""

chat_template = ChatPromptTemplate.from_template(prompt)

for chunk in chatgpt.stream(chat_template.format()):
    print(chunk.content)
    
    

































