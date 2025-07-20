from __future__ import annotations

import json

import boto3
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from infra.llm.base import BaseLLMClient
from shared.logging import get_logger

logger = get_logger(__name__)


class BedrockLLMClient(BaseLLMClient):
    def __init__(self, region: str, model_id: str):
        try:
            self.client = boto3.client('bedrock-runtime', region_name=region)
            self.model_id = model_id
        except NoCredentialsError:
            logger.error('AWS credentials not found')
            raise
        except ClientError as e:
            logger.error(f'AWS client error: {str(e)}')
            raise
        except Exception as e:
            logger.error(f'Error initializing Bedrock LLM: {str(e)}')
            raise

    def generate(self, prompt: str, context: str = '', temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            full_prompt = f'{context}\n\n{prompt}' if context else prompt

            body = {
                'anthropic_version': 'bedrock-2023-05-31',
                'messages': [{'role': 'user', 'content': full_prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens,
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json',
            )

            result = json.loads(response['body'].read())
            return result.get('content', [])[0].get('text', '').strip()
        except Exception as e:
            logger.error(f'Error generating response from Claude: {str(e)}')
            raise
