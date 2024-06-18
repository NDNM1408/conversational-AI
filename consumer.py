import pika
import json
import weaviate
from src.components.embedding.custom.text_embeddings_inference.base import TextEmbeddingsInference

from llama_index.core import Document
from pyvi import ViTokenizer
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter



WEAVIATE_URL = "http://localhost:9090"
DATA_COLLECTION = "Test_Film"
embed_model = TextEmbeddingsInference(base_url="http://127.0.0.1:8081", 
                                      timeout=60)


client = weaviate.Client(WEAVIATE_URL)

vector_store = WeaviateVectorStore(weaviate_client=client,
                                   index_name=DATA_COLLECTION)

storage_context = StorageContext.from_defaults(vector_store=vector_store)



base_node_parser = TokenTextSplitter( 
                                chunk_overlap=0,
                                chunk_size=900,
                                separator=" ",
                                backup_separators=["__", "..", "--"],
                                include_prev_next_rel=False
                                )

child_node_parser= SentenceSplitter(
                    chunk_size=600,
                    chunk_overlap=90,
                    separator=" ",
                    include_prev_next_rel=False,
                    )

def to_string(sample):
    information = ""
    information += f"The name of film is {sample['name']}. "
    information += sample['description']
    genres = ','.join(sample['genre'])
    information += f"The genres of film are {genres}. "
    information += f"The director of film is {sample['director']}. "
    information += f"Some stars of the film are {','.join(sample['stars'])}"
    return information

def on_message_received(ch, method, properties, body):
    print("Received JSON data:", body.decode())
    # Optionally, you can process the JSON data here
    data = json.loads(body)
    print("Processed data:", data)
    documents = []
    documents.append(Document(text=to_string(data),
                            metadata={  
                                "filmname": data["name"],
                            },
                            text_template="{content}"))
    base_nodes = base_node_parser.get_nodes_from_documents(documents)
    child_nodes = child_node_parser.get_nodes_from_documents(base_nodes)
    child_nodes[0].text = ViTokenizer.tokenize(child_nodes[0].text.lower())
    _ = VectorStoreIndex(child_nodes, 
                         storage_context=storage_context, 
                         embed_model=embed_model,
                         insert_batch_size=32768,
                         show_progress=True)
    # Acknowledge the message as processed
    ch.basic_ack(delivery_tag=method.delivery_tag)

# RabbitMQ connection parameters
credentials = pika.PlainCredentials('user', 'password')
parameters = pika.ConnectionParameters('localhost',
                                       5673,
                                       '/',
                                       credentials)

# Establishing connection
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Ensure the queue exists
queue_name = 'json_queue'
channel.queue_declare(queue=queue_name, durable=True)

# Start consuming messages
channel.basic_consume(queue=queue_name,
                      on_message_callback=on_message_received)

print("Starting to consume...")
try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
connection.close()
print("Stopped consuming.")
