import pika
import json
import time

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

# JSON data to send
with open('action.json', 'r') as f:
    full_data = json.load(f)
for data in full_data:
    json_data = json.dumps(data)

    # Publishing the message
    channel.basic_publish(exchange='',
                        routing_key=queue_name,
                        body=json_data,
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # make message persistent
                        ))

    print("Sent JSON data:", json_data)
    time.sleep(1)

# Close the connection
connection.close()

