import os
import openai
import redis
import redisai as rai
from dotenv import load_dotenv
import re
import random
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

quote_set_name = "kant_quotes"
embeddings_hash_name = "kant_quote_embeddings"

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
rai_client = rai.Client(host='localhost', port=6379)


def mock_get_embedding_from_openai(text):
    # Mock function to simulate getting an embedding from OpenAI.
    return np.random.rand(512).astype(np.float32)


def store_embedding_in_redis(quote, embedding):
    # Store the embedding in RedisAI.
    redis_client.hset(embeddings_hash_name, quote, embedding.tobytes())


def is_embedding_unique(new_embedding):
    # Check if the new embedding is unique compared to those in RedisAI.
    for key in redis_client.hkeys(embeddings_hash_name):
        existing_embedding_bytes = redis_client.hget(embeddings_hash_name, key)
        existing_embedding = np.frombuffer(
            existing_embedding_bytes, dtype=np.float32)
        similarity = np.dot(new_embedding, existing_embedding) / \
            (np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))
        if similarity > 0.9:  # Adjust this threshold as needed
            return False
    return True


def get_new_prompt(redis_client):
    previous_quotes = list(quote.decode('utf-8')
                           for quote in redis_client.smembers(quote_set_name))[-5:]
    context = "\\n".join(previous_quotes)
    prompt = f'Based on the following quotes from Immanuel Kant:\\n{context}\\n\\nPlease generate a new and unique quote from Immanuel Kant thats different from the ones above.'
    return prompt


if __name__ == "__main__":
    target_no_items = 100

    while redis_client.scard(quote_set_name) < target_no_items:
        new_prompt = get_new_prompt(redis_client)
        new_quote_data = extract_data(OPENAI_API_KEY, OPENAI_MODEL, new_prompt)
        new_quote = format_data(new_quote_data)[
            0] if format_data(new_quote_data) else None
        new_embedding = mock_get_embedding_from_openai(new_quote)

        if new_quote and is_embedding_unique(new_embedding):
            redis_client.sadd(quote_set_name, new_quote)
            store_embedding_in_redis(new_quote, new_embedding)
            print(f"Added {new_quote} to Redis")

    print(f"{redis_client.scard(quote_set_name)} unique quotes stored in Redis")
